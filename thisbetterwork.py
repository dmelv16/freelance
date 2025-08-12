def classify_voltage_channel(voltage_series: pd.Series):
    """
    Fast classification that robustly identifies the initial settling period and
    locks in the true steady state, while handling temporary dips without
    ending steady state.
    """
    # --- Tunable Parameters ---
    STEADY_VOLTAGE_THRESHOLD = 22.0  # Minimum voltage to be considered for steady state
    STABILITY_WINDOW = 15          # Window for checking if voltage has settled
    
    # NEW: How many standard deviations from the mean defines the steady state band
    STEADY_STATE_TOLERANCE_STD = 3.5 

    # Ramp-down detection parameters
    RAMP_DOWN_THRESHOLD = 1.0      # Voltage drop to consider a ramp-down event
    DIP_RECOVERY_WINDOW = 20       # Points to check for recovery after a dip
    SUSTAINED_DROP_POINTS = 10     # How many points must stay low for true ramp-down
    END_BUFFER_POINTS = 15         # Points at the end to check more carefully for ramp-down

    # Vectorized operations for speed
    voltage = pd.to_numeric(voltage_series, errors='coerce').values
    n = len(voltage)
    
    # Initialize status array
    status = np.full(n, 'De-energized', dtype=object)
    
    # Find energized indices (vectorized)
    energized_mask = (voltage >= 2.2) & (~np.isnan(voltage))
    
    if not np.any(energized_mask):
        return pd.Series(status)
    
    # Get energized boundaries
    energized_indices = np.where(energized_mask)[0]
    start_idx = energized_indices[0]
    end_idx = energized_indices[-1]
    
    # Work only with the energized portion
    energized_voltage = voltage[start_idx:end_idx + 1]
    energized_length = len(energized_voltage)

    if energized_length < STABILITY_WINDOW:
        # Not enough data to classify, mark all as stabilizing
        status[start_idx:end_idx + 1] = 'Stabilizing'
        return pd.Series(status)

    # --- NEW CORE LOGIC: Find the true steady state level first ---
    # We sample the middle of the signal to get a reliable steady state value,
    # avoiding initial transients or final ramp-downs.
    sample_start = int(energized_length * 0.2)
    sample_end = int(energized_length * 0.8)
    
    if sample_end - sample_start < 20: # If signal is too short, use all of it
        sample_start, sample_end = 0, energized_length
        
    steady_state_sample = energized_voltage[sample_start:sample_end]
    valid_sample = steady_state_sample[steady_state_sample >= STEADY_VOLTAGE_THRESHOLD]

    if len(valid_sample) < 10: # If sample is not good, fall back to the full signal
        valid_sample = energized_voltage[energized_voltage >= STEADY_VOLTAGE_THRESHOLD]

    if len(valid_sample) == 0:
        # No points are above the threshold, so it's all stabilizing
        status[start_idx:end_idx + 1] = 'Stabilizing'
        return pd.Series(status)

    # Calculate robust steady state characteristics
    steady_mean = np.mean(valid_sample)
    steady_std = np.std(valid_sample)
    steady_std = max(steady_std, 0.01) # Avoid division by zero or near-zero std

    # Define the tolerance band for steady state
    tolerance = STEADY_STATE_TOLERANCE_STD * steady_std
    upper_bound = steady_mean + tolerance
    lower_bound = steady_mean - tolerance

    # Find the end of the initial stabilizing period
    # This is the first point where the next `STABILITY_WINDOW` points are all within the band
    stabilizing_end_idx = energized_length # Default to all stabilizing
    for i in range(energized_length - STABILITY_WINDOW + 1):
        window = energized_voltage[i:i + STABILITY_WINDOW]
        if np.all((window >= lower_bound) & (window <= upper_bound)):
            stabilizing_end_idx = i
            break
            
    # Set initial statuses
    status[start_idx : start_idx + stabilizing_end_idx] = 'Stabilizing'
    status[start_idx + stabilizing_end_idx : end_idx + 1] = 'Steady State'
    
    # --- RAMP-DOWN DETECTION (Largely Unchanged, but now uses the robust steady_mean) ---
    # Now check for ramp-downs in the main body
    check_start = stabilizing_end_idx + 20 # Start check after initial stabilization
    check_end = max(check_start, energized_length - END_BUFFER_POINTS)

    if check_start < check_end:
        for i in range(check_start, check_end):
            # Only check points currently marked as 'Steady State'
            if status[start_idx + i] == 'Steady State' and (steady_mean - energized_voltage[i] > RAMP_DOWN_THRESHOLD):
                # Potential drop found, look ahead to see if it's a real ramp-down or just a dip
                look_ahead_window = min(DIP_RECOVERY_WINDOW, energized_length - i)
                future_voltages = energized_voltage[i : i + look_ahead_window]
                
                # If it doesn't recover, it's a true ramp-down
                if np.mean(future_voltages) < steady_mean - RAMP_DOWN_THRESHOLD * 0.7:
                    status[start_idx + i : end_idx + 1] = 'Stabilizing' # Mark rest of signal
                    break # Exit the loop once a ramp-down is confirmed

    # Final check: if the very last point is significantly low, find where the drop started
    if energized_voltage[-1] < steady_mean - RAMP_DOWN_THRESHOLD:
        for j in range(energized_length - 1, stabilizing_end_idx, -1):
            if energized_voltage[j] > lower_bound:
                status[start_idx + j + 1 : end_idx + 1] = 'Stabilizing'
                break
                
    return pd.Series(status)


# --- Inside classify_voltage_channel function ---

# (Previous code remains the same up to this point)

# Calculate robust steady state characteristics
steady_mean = np.mean(valid_sample)
steady_std = np.std(valid_sample)

# --- ADD THIS NEW BLOCK ---
# Refinement: To handle highly volatile signals, find the most stable region
# within our sample to calculate a more reliable steady state benchmark.
rolling_std_of_sample = pd.Series(valid_sample).rolling(window=10, min_periods=1).std()
# Use only the calmest 50% of the sample points for our calculation
calm_threshold = rolling_std_of_sample.quantile(0.5) 
stable_points_for_benchmark = valid_sample[rolling_std_of_sample <= calm_threshold]

# Only use this refined sample if it's large enough to be reliable
if len(stable_points_for_benchmark) > 30:
    steady_mean = np.mean(stable_points_for_benchmark)
    steady_std = np.std(stable_points_for_benchmark)
# --- END OF NEW BLOCK ---

steady_std = max(steady_std, 0.01) # Avoid division by zero or near-zero std

# (The rest of the function remains the same)
