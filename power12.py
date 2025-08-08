def classify_voltage_channel(voltage_series: pd.Series):
    """
    Fast classification that properly identifies stabilizing periods, steady states,
    handles gaps in data, and accounts for step changes in steady state voltage.
    """
    # --- Tunable Parameters ---
    STEADY_VOLTAGE_THRESHOLD = 22
    STABILITY_WINDOW = 10  # Window for checking stability
    STABILITY_THRESHOLD = 0.1  # Max std dev for steady state
    STABILIZING_MAX_DURATION = 0.1  # Max 10% of test can be stabilizing
    
    # Gap detection
    GAP_VOLTAGE_THRESHOLD = 2.2  # Below this is considered de-energized
    MIN_GAP_LENGTH = 5  # Minimum points to consider it a true gap
    
    # Step change detection
    STEP_CHANGE_THRESHOLD = 1.5  # Voltage change to consider a step
    STEP_STABILITY_WINDOW = 20  # Points to check if new level is stable
    
    # Ramp-down detection parameters
    RAMP_DOWN_THRESHOLD = 1.0  # Voltage drop to consider ramp-down
    DIP_RECOVERY_WINDOW = 20  # Points to check for recovery after dip
    SUSTAINED_DROP_POINTS = 10  # How many points must stay low for true ramp-down
    END_BUFFER_POINTS = 10  # Points at end to check more carefully
    
    # Vectorized operations for speed
    voltage = pd.to_numeric(voltage_series, errors='coerce').values
    n = len(voltage)
    
    # Initialize status array
    status = np.full(n, 'De-energized', dtype=object)
    
    # Find energized indices
    energized_mask = (voltage >= GAP_VOLTAGE_THRESHOLD) & (~np.isnan(voltage))
    
    if not np.any(energized_mask):
        return pd.Series(status)
    
    # Find continuous energized segments (handling gaps)
    energized_segments = []
    in_segment = False
    segment_start = None
    
    for i in range(n):
        if energized_mask[i]:
            if not in_segment:
                segment_start = i
                in_segment = True
        else:
            if in_segment:
                # Check if this is a real gap or just a single point dropout
                gap_end = i
                for j in range(i, min(i + MIN_GAP_LENGTH, n)):
                    if energized_mask[j]:
                        gap_end = j
                        break
                
                if gap_end - i >= MIN_GAP_LENGTH or gap_end == i:
                    # This is a real gap, close the segment
                    energized_segments.append((segment_start, i - 1))
                    in_segment = False
    
    # Close final segment if needed
    if in_segment:
        energized_segments.append((segment_start, n - 1))
    
    # Process each energized segment independently
    for seg_start, seg_end in energized_segments:
        segment_voltage = voltage[seg_start:seg_end + 1]
        segment_length = len(segment_voltage)
        
        # Calculate rolling statistics for this segment
        rolling_std = pd.Series(segment_voltage).rolling(
            window=STABILITY_WINDOW, min_periods=1
        ).std().fillna(method='bfill').values
        
        # Find where voltage is stable
        stable_mask = (segment_voltage >= STEADY_VOLTAGE_THRESHOLD) & (rolling_std < STABILITY_THRESHOLD)
        stable_indices = np.where(stable_mask)[0]
        
        if len(stable_indices) > 0:
            first_stable = stable_indices[0]
            
            # Limit stabilizing period
            max_stabilizing = int(segment_length * STABILIZING_MAX_DURATION)
            stabilizing_end = min(first_stable, max_stabilizing)
            
            # Set initial stabilizing period
            if stabilizing_end > 0:
                status[seg_start:seg_start + stabilizing_end] = 'Stabilizing'
            
            # Track current steady state level
            current_ss_start = stabilizing_end
            steady_sample_end = min(stabilizing_end + 100, segment_length)
            steady_sample = segment_voltage[stabilizing_end:steady_sample_end]
            current_steady_mean = np.mean(steady_sample[steady_sample >= STEADY_VOLTAGE_THRESHOLD])
            current_steady_std = np.std(steady_sample[steady_sample >= STEADY_VOLTAGE_THRESHOLD])
            
            # Process the rest of the segment
            i = stabilizing_end
            while i < segment_length:
                # Check for step changes in steady state
                if i > current_ss_start + STEP_STABILITY_WINDOW:
                    recent_window = segment_voltage[i-STEP_STABILITY_WINDOW:i]
                    recent_mean = np.mean(recent_window)
                    recent_std = np.std(recent_window)
                    
                    # Check if we've had a step change to a new steady level
                    if (abs(recent_mean - current_steady_mean) > STEP_CHANGE_THRESHOLD and
                        recent_std < STABILITY_THRESHOLD * 2):
                        # New steady state level detected
                        current_steady_mean = recent_mean
                        current_steady_std = recent_std
                        current_ss_start = i - STEP_STABILITY_WINDOW
                        # Mark this as steady state (new level)
                        status[seg_start + current_ss_start:seg_start + i] = 'Steady State'
                
                # Check for ramp-down at end of segment
                if i >= segment_length - END_BUFFER_POINTS:
                    end_voltages = segment_voltage[i:]
                    end_mean = np.mean(end_voltages)
                    
                    if len(end_voltages) >= 3:
                        end_trend = np.polyfit(np.arange(len(end_voltages)), end_voltages, 1)[0]
                    else:
                        end_trend = 0
                    
                    # Check for end ramp-down
                    if (current_steady_mean - end_mean > RAMP_DOWN_THRESHOLD * 0.7 or 
                        end_trend < -0.05 or
                        np.sum(end_voltages < current_steady_mean - RAMP_DOWN_THRESHOLD) > len(end_voltages) * 0.5):
                        status[seg_start + i:seg_end + 1] = 'Stabilizing'
                        break
                    else:
                        status[seg_start + i] = 'Steady State'
                
                # Check for mid-segment ramp-down
                elif i > current_ss_start + 50:
                    current_voltage = segment_voltage[i]
                    
                    if current_steady_mean - current_voltage > RAMP_DOWN_THRESHOLD:
                        # Look ahead for recovery
                        look_ahead_window = min(DIP_RECOVERY_WINDOW, segment_length - i)
                        future_voltages = segment_voltage[i:i + look_ahead_window]
                        
                        # Check if voltage recovers
                        recovery_mask = future_voltages > (current_steady_mean - RAMP_DOWN_THRESHOLD * 0.5)
                        if np.sum(recovery_mask) > look_ahead_window * 0.3:
                            # Temporary dip - stays steady state
                            status[seg_start + i] = 'Steady State'
                        else:
                            # Check if drop is sustained
                            sustained_window = min(SUSTAINED_DROP_POINTS, segment_length - i)
                            sustained_voltages = segment_voltage[i:i + sustained_window]
                            
                            if np.mean(sustained_voltages) < current_steady_mean - RAMP_DOWN_THRESHOLD * 0.7:
                                # True ramp-down
                                status[seg_start + i:seg_end + 1] = 'Stabilizing'
                                break
                            else:
                                status[seg_start + i] = 'Steady State'
                    else:
                        status[seg_start + i] = 'Steady State'
                else:
                    # Normal steady state
                    status[seg_start + i] = 'Steady State'
                
                i += 1
                
        else:
            # No stable points found in this segment - all stabilizing
            status[seg_start:seg_end + 1] = 'Stabilizing'
    
    return pd.Series(status)


def clean_dc_channels(group_df):
    """
    Fast orchestrator for cleaning DC channels with new data structure.
    Expects columns: 'Value' (voltage) and 'DC' (DC1 or DC2)
    """
    df = group_df.copy()
    
    # Initialize status column
    df['status'] = 'De-energized'
    
    # Check if DC column exists
    if 'DC' not in df.columns:
        # If no DC column, assume all data is DC1
        df['status'] = classify_voltage_channel(df['Value'])
        return df
    
    # Process each DC channel separately
    for dc_channel in df['DC'].unique():
        # Get mask for this DC channel
        dc_mask = df['DC'] == dc_channel
        
        # Process voltage data for this channel
        if dc_mask.sum() > 0:
            channel_status = classify_voltage_channel(df.loc[dc_mask, 'Value'])
            df.loc[dc_mask, 'status'] = channel_status.values
    
    return df
