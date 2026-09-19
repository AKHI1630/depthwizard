"""Verify pysolar azimuth convention with hand-checkable cases."""
from pysolar.solar import get_altitude, get_azimuth
from datetime import datetime, timezone

# Test 1: Solar noon on equinox at 28N
# Sun should be due south -> compass azimuth 180 deg
dt_noon = datetime(2024, 3, 20, 6, 20, 0, tzinfo=timezone.utc)
elev = get_altitude(28.0, 85.0, dt_noon)
az_raw = get_azimuth(28.0, 85.0, dt_noon)
print("Test 1 — equinox noon at 28N 85E:")
print(f"  get_azimuth raw: {az_raw:.2f}")
print(f"  get_altitude:    {elev:.2f}")
print(f"  Expected: sun due south -> compass 180 deg")
print(f"  If raw ~180 -> already compass (N=0)")
print(f"  If raw ~0   -> from-south convention")
print()

# Test 2: Morning sun (08:00 local = 02:15 UTC)
dt_morn = datetime(2024, 3, 20, 2, 15, 0, tzinfo=timezone.utc)
elev2 = get_altitude(28.0, 85.0, dt_morn)
az_raw2 = get_azimuth(28.0, 85.0, dt_morn)
print("Test 2 — equinox 08:00 local at 28N 85E:")
print(f"  get_azimuth raw: {az_raw2:.2f}")
print(f"  get_altitude:    {elev2:.2f}")
print(f"  Expected: morning sun in east -> compass ~90 deg")
print()

# Test 3: The actual WV02 acquisition
dt_wv = datetime(2024, 5, 29, 5, 10, 55, tzinfo=timezone.utc)
elev3 = get_altitude(27.744, 85.322, dt_wv)
az_raw3 = get_azimuth(27.744, 85.322, dt_wv)
print("Test 3 — WV02 acquisition 05:10:55Z (10:55 local):")
print(f"  get_azimuth raw: {az_raw3:.2f}")
print(f"  get_altitude:    {elev3:.2f}")
print(f"  Expected: late morning sun -> compass ~105-115 deg")
print()

# Conclusion
print("=" * 50)
if abs(az_raw - 180) < 20:
    correct_wv = az_raw3
    print("pysolar returns COMPASS bearing (N=0, CW)")
    print("DO NOT add 180 -- the current code is WRONG")
    print(f"WV02 correct azimuth: {correct_wv:.2f} deg")
elif abs(az_raw) < 20 or abs(az_raw - 360) < 20:
    correct_wv = (az_raw3 + 180) % 360
    print("pysolar returns from-south bearing")
    print("Add 180 to get compass -- current code is CORRECT")
    print(f"WV02 correct azimuth: {correct_wv:.2f} deg")
else:
    print(f"UNCLEAR: noon raw = {az_raw:.1f}")
    print("Need further investigation")
