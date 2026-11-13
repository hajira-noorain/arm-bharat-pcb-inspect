#!/usr/bin/env bash
# Turn the Pi's Wi-Fi into an offline hotspot so a phone can open the dashboard with no internet.
#   bash deploy/hotspot.sh [ssid] [password]
# Then on the phone: join the network and open http://10.42.0.1:8000
set -euo pipefail
SSID="${1:-PCB-Station}"
PASS="${2:-inspect1234}"
sudo nmcli device wifi hotspot ifname wlan0 ssid "$SSID" password "$PASS"
nmcli -g IP4.ADDRESS device show wlan0
