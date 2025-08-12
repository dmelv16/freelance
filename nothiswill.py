def find_stabilizing_elbow(voltage_data: np.ndarray, ramp_window=15, flat_window=15, flat_slope_threshold=0.01):
    """
    Finds the "elbow" where the signal transitions from a ramp-up to a steady state.
    """
    max_slope_change = -1
    best_elbow_index = -1
    search_range = len(voltage_data) - flat_window
    x_ramp, x_flat = np.arange(ramp_window), np.arange(flat_window)

    for i in range(ramp_window, search_range):
        ramp_segment = voltage_data[i - ramp_window : i]
        flat_segment = voltage_data[i : i + flat_window]
        ramp_slope = np.polyfit(x_ramp, ramp_segment, 1)[0]
        flat_slope = np.polyfit(x_flat, flat_segment, 1)[0]
        
        if ramp_slope > flat_slope and abs(flat_slope) < flat_slope_threshold:
            slope_change = ramp_slope - flat_slope
            if slope_change > max_slope_change:
                max_slope_change = slope_change
                best_elbow_index = i
                
    return best_elbow_index

def classify_voltage_channel(voltage_series: pd.Series):
    """
    Hybrid classifier that uses a geometric elbow-finder for precision and a 
    statistical method for robustness, automatically choosing the best approach.
    """
    # --- Tunable Parameters ---
    STEADY_VOLTAGE_THRESHOLD = 22.0
    STABILITY_WINDOW = 15
    STEADY_STATE_TOLERANCE_STD = 3.5
    RAMP_DOWN_THRESHOLD = 1.0
    DIP_RECOVERY_WINDOW = 20
    END_BUFFER_POINTS = 15

    # --- Setup ---
    voltage = pd.to_numeric(voltage_series, errors='coerce').values
    n = len(voltage)
    status = np.full(n, 'De-energized', dtype=object)
    
    energized_mask = (voltage >= 2.2) & (~np.isnan(voltage))
    if not np.any(energized_mask):
        return pd.Series(status)
        
    energized_indices = np.where(energized_mask)[0]
    start_idx, end_idx = energized_indices[0], energized_indices[-1]
    energized_voltage = voltage[start_idx : end_idx + 1]
    energized_length = len(energized_voltage)

    if energized_length < 30:
        status[start_idx : end_idx + 1] = 'Stabilizing'
        return pd.Series(status)

    # --- HYBRID LOGIC: CALCULATE BOTH WAYS ---

    # 1. The Robust Statistical Method (The "Generalist")
    # First, get a reliable steady-state baseline
    sample_start = int(energized_length * 0.2)
    sample_end = int(energized_length * 0.8)
    if sample_end - sample_start < 20:
        sample_start, sample_end = 0, energized_length
    
    valid_sample = energized_voltage[sample_start:sample_end]
    valid_sample = valid_sample[valid_sample >= STEADY_VOLTAGE_THRESHOLD]
    
    if len(valid_sample) < 10:
        status[start_idx : end_idx + 1] = 'Stabilizing'
        return pd.Series(status)

    steady_mean = np.mean(valid_sample)
    steady_std = max(np.std(valid_sample), 0.01)
    
    # Find stabilization end using the tolerance band
    tolerance = STEADY_STATE_TOLERANCE_STD * steady_std
    upper_bound = steady_mean + tolerance
    lower_bound = steady_mean - tolerance
    
    stabilizing_end_idx_stat = energized_length # Default
    for i in range(energized_length - STABILITY_WINDOW + 1):
        window = energized_voltage[i : i + STABILITY_WINDOW]
        if np.all((window >= lower_bound) & (window <= upper_bound)):
            stabilizing_end_idx_stat = i
            break

    # 2. The Precise Geometric Method (The "Specialist")
    # Search for a clean elbow in the first part of the signal
    search_len = min(1000, int(energized_length * 0.4))
    elbow_idx_geom = find_stabilizing_elbow(energized_voltage[:search_len])

    # --- DECISION ENGINE: CHOOSE THE BEST RESULT ---
    final_stabilizing_end_idx = 0
    if elbow_idx_geom != -1:
        # An elbow was found. We must verify it's a genuine ramp-up to trust it.
        # Check that the voltage before the elbow is significantly lower than after.
        mean_before = np.mean(energized_voltage[max(0, elbow_idx_geom - 10) : elbow_idx_geom])
        mean_after = np.mean(energized_voltage[elbow_idx_geom : elbow_idx_geom + 10])

        # Use the precise geometric elbow ONLY if it's a clear ramp-up.
        if mean_before < mean_after - 0.5: # -0.5V threshold confirms it's a real climb
            final_stabilizing_end_idx = elbow_idx_geom
        else:
            # The "elbow" wasn't a ramp-up (e.g., a settle-down).
            # The statistical method is more reliable for this shape.
            final_stabilizing_end_idx = stabilizing_end_idx_stat
    else:
        # No geometric elbow was found at all. We must use the robust statistical result.
        final_stabilizing_end_idx = stabilizing_end_idx_stat
        
    # --- FINAL CLASSIFICATION ---
    status[start_idx : start_idx + final_stabilizing_end_idx] = 'Stabilizing'
    status[start_idx + final_stabilizing_end_idx : end_idx + 1] = 'Steady State'
    
    # --- RAMP-DOWN DETECTION (Uses the robust steady_mean calculated earlier) ---
    check_start = final_stabilizing_end_idx + 20
    check_end = max(check_start, energized_length - END_BUFFER_POINTS)
    if check_start < check_end:
        for i in range(check_start, check_end):
            if status[start_idx + i] == 'Steady State' and (steady_mean - energized_voltage[i] > RAMP_DOWN_THRESHOLD):
                look_ahead_window = min(DIP_RECOVERY_WINDOW, energized_length - i)
                if np.mean(energized_voltage[i : i + look_ahead_window]) < steady_mean - RAMP_DOWN_THRESHOLD * 0.7:
                    status[start_idx + i : end_idx + 1] = 'Stabilizing'
                    break

    return pd.Series(status)
