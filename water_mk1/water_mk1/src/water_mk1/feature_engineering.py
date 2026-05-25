import pandas as pd
from pathlib import Path

OPTIMAL_MIN = 1.2  # MPa
OPTIMAL_MAX = 1.4  # MPa


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["ts"])
    return df.sort_values("ts").reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:

    DIAGNOSTIC_COLS = [
        "lsid", "sensor_type", "station_id",
        "noise_floor_rssi", "port_energy_consumed_4", "port_energy_consumed_3",
        "port_energy_consumed_2", "port_energy_consumed_1", "beacon_interval",
        "door_switch_status", "link_layer_packets_received", "solar_panel_voltage",
        "rank", "health_version", "false_wakeup_rssi", "inside_box_temp",
        "dcell_battery_voltage", "power_percentage_tx", "last_rx_rssi",
        "power_percentage_mcu", "application_firmware_version", "false_wakeup_count",
        "etx", "rpl_mode", "uptime", "number_of_neighbors", "last_parent_rtt_ping",
        "li_ion_cell_voltage", "platform_id", "power_percentage_rx",
        "rpl_parent_node_id", "access_technology", "lead_acid_battery_voltage",
        "location_area_code", "creg_cgreg", "cell_id", "rssi", "mcc_mnc", "cme",
        "arch_int", "wind_num_samples", "iss_reception", "rev_type",
        "rainfall_clicks", "rain_rate_hi_clicks", "tz_offset",
    ]

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    keep = [c for c in numeric_cols if c not in DIAGNOSTIC_COLS]
    df = df.groupby("ts")[keep].mean().reset_index()

    # ── 1. Rimuovi outlier water_potential (range fisicamente plausibile per la vite)
    df["water_potential_last"] = df["water_potential_last"].clip(lower=-15.0, upper=-0.1)

    # ── 2. Interpolazione temporale per i NaN
    df = df.set_index("ts").sort_index()
    df["water_potential_last"] = (
        df["water_potential_last"]
        .interpolate(method="time", limit=6)   # max 6 ore consecutive
        .ffill(limit=3)                         # fallback forward fill
        .bfill(limit=3)                         # fallback backward fill
    )
    df["moist_soil_last_1"] = (
        df["moist_soil_last_1"]
        .interpolate(method="time", limit=6)
        .ffill(limit=3)
        .bfill(limit=3)
    )
    df = df.reset_index()

    # ── 3. Rinomina
    df = df.rename(columns={
        "water_potential_last": "water_potential_avg",
        "moist_soil_last_1":    "moist_soil_avg",
        "temp_out":             "temp_out_avg",
        "solar_rad_avg":        "solar_rad_avg_val",
        "wind_speed_avg":       "wind_speed_val",
        "hum_out":              "hum_out_avg",
        "et":                   "et_avg",
    })

    # ── 4. Pioggia cumulata 24h
    if "rainfall_mm" in df.columns:
        df["rain_24h"] = df["rainfall_mm"].rolling(window=24, min_periods=1).sum()
    else:
        df["rain_24h"] = 0.0

    # ── 5. Label
    df["irrigate"] = df["water_potential_avg"] < OPTIMAL_MIN

    return df


def get_feature_columns() -> list:
    return [
        "water_potential_avg",
        "moist_soil_avg",
        "rain_24h",
        "temp_out_avg",
        "solar_rad_avg_val",
        "wind_speed_val",
        "hum_out_avg",
        "et_avg",
    ]
