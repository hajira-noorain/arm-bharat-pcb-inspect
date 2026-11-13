"""The Pi 5 power reading: parsing the power chip's rail table (no Pi needed for the arithmetic)."""
from bench.bench_device import PowerLog, parse_pmic

SAMPLE = """ 3V7_WL_SW_A current(0)=0.10000000A
 3V3_SYS_A current(1)=0.50000000A
 VDD_CORE_A current(7)=2.00000000A
 3V7_WL_SW_V volt(8)=3.70000000V
 3V3_SYS_V volt(9)=3.30000000V
 VDD_CORE_V volt(15)=0.80000000V
 HDMI_V volt(16)=5.10000000V
 EXT5V_V volt(24)=5.12000000V
"""


def test_power_is_the_sum_of_voltage_times_current_over_rails_that_report_both():
    assert parse_pmic(SAMPLE) == round(0.1 * 3.7 + 0.5 * 3.3 + 2.0 * 0.8, 3)     # HDMI / EXT5V have no current


def test_no_reading_off_the_pi_is_reported_as_missing_not_as_zero():
    assert parse_pmic("") is None and parse_pmic("command not found") is None
    with PowerLog(period=0.01) as pw:
        pass
    assert pw.summary() is None
