Subject: <SEVERITY> <device_id>: <short symptom>

DEVICE     <device_id> (<model>), mode <operating_mode>
SEVERITY   <P1 | P2 | requested by operator>
READING    temp <chip_temperature_c> C | hashrate <asic_hashrate_th> TH/s |
           power <asic_power_w> W | errors <error_count> | fan <fan_speed_rpm> rpm
TAKEN AT   <timestamp of the reading>

WHAT WE SEE
<one or two sentences: the values that crossed a threshold and which threshold>

PROBABLE CAUSE
<failure mode, with confidence: high | medium | low> | not diagnosed

RECOMMENDED ACTION (advisory)
<action> - executed by <Decision Engine | on-site technician>

Sent by the MDK maintenance agent. Numbers come from the ML layer at the time shown above.
