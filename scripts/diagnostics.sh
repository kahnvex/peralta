#!/bin/bash

set -e

VPN_IP=$1

echo "Running networkQuality"
networkQuality

echo "Running traceroute"
traceroute -q 1 -m 15 $VPN_IP

