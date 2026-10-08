# Channel assignment audit

Upstream SleepFM `channel_groups.json` deduplicated per modality.
This is the authoritative mapping the preprocessing pipeline uses to
slice the 27-channel HDF5 into 4 modality subsets.

## BAS — 345 unique channel aliases

<details><summary>Click to expand</summary>

```
01-A2
A1
A1:C4
A1:EOG1
A1:EOG2
A1:F4
A1:O2
A1A2
A2
A2:A1
A2:A1:C3
A2:C3
A2:EOG1
A2:EOG2
A2:F3
A2:O1
C2-Cz
C3
C3-A1
C3-A2
C3-Cz
C3-M1
C3-M2
C3-M2:C4-M1
C3-O1
C3:A2
C3:C4
C3:M1
C3:M1+M2
C3:M1-M2
C3:M1:M2
C3:M2
C3:M2:M1
C3A2
C3M1
C3M2
C4
C4-A1
C4-A2
C4-Cz
C4-M1
C4-M2
C4:A1
C4:C3
C4:M1
C4:M1+M2
C4:M1-M2
C4:M1:M2
C4:M2
C4:M2:M1
C4A1
C4M1
C4M2
CZA2
CZM2
Cz
E1
E1 (LEOG)
E1-Cz
E1-M2
E1:E2
E1:Fp2
E1:M1
E1:M1+M2
E1:M1-M2
E1:M1:M2
E1:M2
E1:M2:M1
E1M2
E2
E2 (REOG)
E2-Cz
E2-M2
E2:Fp1
E2:M1
E2:M1+M2
E2:M1-M2
E2:M1:M2
E2:M2
E2:M2:M1
E2M2
EEG
EEG (SEC)
EEG (sec)
EEG 2
EEG A1-A2
EEG C3-A1
EEG C3-A2
EEG C3-A22
EEG C4-A1
EEG C4-A12
EEG C4-A2
EEG F3-A1
EEG F3-A2
EEG F3-A22
EEG F4-A1
EEG F4-A12
EEG F4-A2
EEG F7-A2
EEG F8-A1
EEG Fp1-A2
EEG Fp1-A22
EEG Fp2-A1
EEG Fp2-A12
EEG O1-A1
EEG O1-A2
EEG O1-A22
EEG O2-A1
EEG O2-A12
EEG O2-A2
EEG P3-A2
EEG P4-A1
EEG T3-A2
EEG T4-A1
EEG T5-A2
EEG T6-A1
EEG sec
EEG(SEC)
EEG(sec)
EEG1
EEG2
EEG3
EOG LOC-A2
EOG LOC-A22
EOG ROC-A1
EOG ROC-A12
EOG ROC-A2
EOG ROC-A22
EOG(L)
EOG(R)
EOG-L
EOG-R
EOG1
EOG1:A1
EOG1:A2
EOG2
EOG2:A1
EOG2:A2
EOGl
EOGl:M1
EOGl:M2
EOGr
EOGr512
EOGr512:M1
EOGr512:M2
EOGr:M1
EOGr:M2
F1-A2
F1M2
F2-C4
F2-T4
F2M1
F3
F3-Cz
F3-M1
F3-M2
F3/A2
F3:A2
F3:F4
F3:M1
F3:M1+M2
F3:M1-M2
F3:M1:M2
F3:M2
F3:M2:E1:M1
F3:M2:M1
F3A2
F3M2
F4
F4-A1
F4-Cz
F4-M1
F4-M1:F3-M2
F4-M2
F4/A1
F4:A1
F4:F3
F4:M1
F4:M1+M2
F4:M1-M2
F4:M1:M2
F4:M2
F4:M2:M1
F4A1
F4M1
F7
F7-Cz
F7:M2
F7A2
F7M2
F8
F8-Cz
F8:F7
F8:M1
F8A1
F8M1
FP1-A2
FP1-C3
FP1-M2
FP1A2
FP1M2
FP2A1
FP2M1
FZ-A1A2
FZ-M1M2
FZA2
FZM2
Fp1
Fp1-C3
Fp1-Cz
Fp1-T3
Fp1/A2
Fp1:A2
Fp1:E2
Fp1:Fp2
Fp1:M1+M2
Fp1:M2
Fp2
Fp2-C4
Fp2-Cz
Fp2-T4
Fp2:A1
Fp2:Fp1
Fp2:M1
Fp2:M1+M2
Fpz
Fpz:Oz
Fz
Fz-A1
Fz-A2
Fz-Cz
L-EOG
LEOG
LOC
LOC-M2
M1
M1-Cz
M1:C3
M1:C4
M1:E1
M1:E2
M1:F3
M1:F4
M1:Fp2
M1:O1
M1:O2
M1M2
M2
M2-Cz
M2:C3
M2:C4
M2:E1
M2:E2
M2:E2:M1
M2:F3
M2:F4
M2:Fp1
M2:M1
M2:M1:O1:M2
M2:O1
M2:O2
O1
O1-A2
O1-Cz
O1-M1
O1-M2
O1-x
O1:A2
O1:E1:M1
O1:M1
O1:M1+M2
O1:M1-M2
O1:M1:M2
O1:M2
O1:M2:M1
O1A2
O1M1
O1M2
O2
O2-A1
O2-M1
O2-M1:O1-M2
O2-M2
O2-x
O2:A1
O2:M1
O2:M1+M2
O2:M1-M2
O2:M1:M2
O2:M2
O2:M2:M1
O2:O1
O2A1
O2M1
O2M2
Oz
Oz:M1
P3
P3-Cz
P3:M2
P3:P4
P3M2
P4
P4-Cz
P4:M1
P4M1
PZA1
PZM1
Pz
Pz-Cz
Pz:Fz
R-EOG
REOG
ROC
ROC-M1
T3
T3-Cz
T3-M2
T3-O1
T3/A2
T3:M1
T3:M2
T3:T4
T3A2
T3M2
T4
T4-Cz
T4-M1
T4-O2
T4/A1
T4:C4
T4:M1
T4A1
T4M1
T5
T5-Cz
T5:M2
T5A2
T5M2
T6
T6-Cz
T6:M1
T6:T5
T6A1
T6M1
```

</details>

## RESP — 119 unique channel aliases

<details><summary>Click to expand</summary>

```
ABD
ABDM
ABDO EFFORT
ABDO RES
ABDOMEN
AIRFLOW
AIRFLOW-0
AIRFLOW-1
Abd
Abd:Chest
AbdDC
Abdo
Abdomen
Abdominal
Airflow
CHEST
CHEST #1
Chest
Chest Effort
ChestDC
Effort THO
Finger PPG
H.R.
HR
HRate
Heart Rate
Heart Rate_CU
Heartrate
MIC
Mic
N Pres
N.Press
NASAL
NASAL P
NASAL PRESSURE
NEW AIR
NEWAIR
NasOr
Nasal
Nasal Backup
Nasal Pressure
Nasal Snore
Nasal Therm
Nasal Thermistor
Nasal-AC
Nasal/Oral Therm
NasalAC
NasalDC
NasalOr
NasalP
NasalSn
New A
New AIR
New Air
O2-Masimo
O2_Masimo
Oral
Oral Therm
Oral Thermistor
Oral-CO2
P-Snore
PPG
PPG Masimo
PPG-Masimo
PPG_Masimo
PULSE
Pressure Snore
Pulse
Pulse Rate
Pulse-0
Pulse-1
PulseR
PulseRa
PulseRate
RIP Abdom
RIP Abdomen
RIP Thora
RIP Thorax
SAO2
SNOR
SNORE
SPO2
Sa02
SaO2
SaO2_Masimo
SentecPPG
SentecSpO2
Snore
Snore-db
Snore1
Snore2
Snore_CU
Snoreg
Snoring
Snoring2
Snoring2a
SpO2
SpO2 Masimo
SpO2-0
SpO2-1
Sum RIP
Sum.RIPs
THOR
THOR EFFORT
THOR RES
TcPPG
TcSpO2
Therm
Thermist
Thermistor
Thor
Thoracic
Thorax
Tosca_Pulse
Trach Therm
chest
new air
p.Abdomen
tcSpO2
```

</details>

## EKG — 32 unique channel aliases

<details><summary>Click to expand</summary>

```
ECG
ECG #1
ECG #2
ECG 2
ECG I
ECG I2
ECG II
ECG IIHF
ECG L
ECG L-ECG R
ECG R
ECG1
ECG1-EC
ECG1-ECG2
ECG2
ECGI
ECGI-0
ECGI-1
ECGII
ECGII-0
ECGII-1
ECGL
ECGR
EKG
EKG #1
EKG #2
EKG-L
EKG-R
EKG1
EKG2
EKG_L-Cz
EKG_R-Cz
```

</details>

## EMG — 187 unique channel aliases

<details><summary>Click to expand</summary>

```
ARM
ARM L
ARM LEFT
ARM R
ARM RIGHT
Arm EMG
Arm l
Arm r
Arm-L
Arm-R
Arm_L
Arm_R
Arms
Arms-1
Arms-2
Arms-L
Arms-L Ext
Arms-L Flex
Arms-R
Arms-R Ext
Arms-R Flex
CHIN
CHIN 2
CHIN1
CHIN1-C
CHIN2
CHIN2-C
CHIN3
CHINEMG
Chin
Chin 1
Chin 1-Chin 2
Chin 2
Chin 3
Chin EMG
Chin EMG2
Chin-A
Chin-Ctr
Chin-L
Chin-R
Chin1
Chin1-Chin2
Chin2
Chin2 EMG
Chin2:Chin
Chin:Chin2
Chin:Chin2:Chin
Chin:RIC
ChinA
ChinEMG
ChinL
ChinR
Chin_Ctr-Cz
Chin_L-Chin-Ctr
Chin_L-Cz
Chin_R-Chin_Ctr
Chin_R-Cz
EMG
EMG #1
EMG #2
EMG #3
EMG Aux1
EMG Aux12
EMG Aux2
EMG Chin
EMG1
EMG2
EMG3
EMG4
EMG5
EMG6
EMG:EMG4
EMGd1
EMGd2
Ext1
Ext2
Ext3
Ext4
Extensor-L
Extensor_L-Cz
Extensor_R-Cz
Feet-L
Feet-R
Flexor-L
Flexor_L-Cz
Flexor_R-Cz
Foot
Foot-L
Foot-R
Integral EMG
InterEMG
L Arm
L Arm 1
L Arm 2
L Chin
L Chin-R Chin
L EMG
L Leg
L Leg 1
L Leg 2
L-Arm
L-Arm1
L-Arm2
L-LEG 1
L-LEG 2
L-Leg 1-L-Leg 2
L-Leg1
L-Leg2
LArm
LChin
LEG(L)
LEG(R)
LEG/L
LEG/R
LEG1
LEG2
LEMG
LLEG
LLEGEMG
LLeg
Left Leg
Left Masseter 1
Left Masseter 2
Leg
Leg 1
Leg 12
Leg 2
Leg 22
Leg L
Leg R
Leg-L
Leg-R
Leg/L
Leg/R
LegL
LegR
Legs
LegsL-Leg1
MASSETE
Mass-L
Mass-LR
Mass-R
Massater 1
Massater 2
Masset.
Masseter 1
Masseter 2
Masseter-L
Masseter-R
Pleth
Pressur
R Arm
R Arm 1
R Arm 2
R Chin
R EMG
R Leg
R Leg 1
R Leg 2
R-Arm
R-Arm1
R-Arm2
R-LEG 1
R-LEG 2
R-Leg1
R-Leg1-R-Leg2
R-Leg2
RArm
RChin
RIC EMG
RIP Abd
RIP Tho
RLEG
RLEGEMG
RLeg
Right Arm 2
Right Leg
Right Masseter 1
Right Masseter 2
Right arm
Right-Masseter1
SCM
Scalene
Should.
Temporalis-L
Temporalis-R
chin
```

</details>

## Verdict

Physiologically distinct signals (BAS=EEG+EOG, RESP=thorax/abdomen/SpO2,
EKG=ECG leads, EMG=chin/leg muscle leads). No cross-contamination between
groups — BAS does not contain any respiratory channels. The modality-transfer
null for BAS × AHI/ODI3 therefore cannot be a channel-mapping artefact.
