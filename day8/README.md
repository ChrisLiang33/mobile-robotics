# Day 8 — Gait Wearable (UNO Q + MPU-6050)

*ME193 — Chris Liang*

An ankle-worn UNO Q with an MPU-6050 six-axis IMU that tells **standing,
walking, running and biking** apart, with two models that use no neural nets:

- **Model 1** looks at one number: the **peak acceleration along one axis**
  (`ax`) in a 2-second window, measured from the window's mean so gravity
  drops out, and splits it with three learned thresholds (a depth-2
  decision tree).
- **Model 2** uses **all six IMU outputs**, combined into the magnitudes of
  the accel and gyro vectors: 8 features per window (mean, std, range of
  each magnitude, plus step frequency and its strength) into a depth-4
  decision tree.

The wearable classifies on the board every half second and shows the activity
as an icon on its LED matrix; the laptop shows both models side by side.

## Files

| File | What it is |
|------|------------|
| `unoq/gait-wearable/` | UNO Q App Lab project: the sketch reads the MPU-6050 at 50 Hz; `main.py` streams it over MQTT, runs `models.json`, draws the icon |
| `record.py` | laptop: record labeled IMU data over MQTT into `data/gait.csv` (press s/w/r/b for what you're doing) |
| `train.py` | laptop: windows → features → both decision trees, leave-one-session-out accuracy, `models.json`, plots |
| `live.py` | laptop: live dashboard with both models' answers and the raw signal |
| `gait_features.py` | the features, the window logic and a numpy-only tree runtime shared by laptop and board |
| `models.json`, `plots/` | the trained models and the pictures for the Notion page (after `train.py`) |

## Wiring

MPU-6050 module → UNO Q: `VCC → 3.3V`, `GND → GND`, `SDA → SDA`, `SCL → SCL`
(the two pins beside AREF). Leave `AD0` open (address 0x68). Strap the UNO Q +
power bank to the **ankle** (a sock over it works); the sensor orientation
doesn't matter for model 2, and for model 1 pick the axis with the biggest
swing with `train.py --axis`.

## Steps

```bash
cd unoq && UNOQ_HOST=<board-ip> ./unoq.sh deploy      # 1. app onto the board (once)
python record.py --session chris_1                    # 2. ~60 s of each activity, press s/w/r/b
python record.py --session chris_2                    #    a second session on another day/person
python train.py                                       # 3. accuracy, models.json, plots/
cd unoq && UNOQ_HOST=<board-ip> ./unoq.sh deploy      # 4. redeploy: now the board has models.json
python live.py                                        # 5. the demo
```

## How it works

Every 20 ms the microcontroller reads the IMU and hands the sample to the
Linux side. There, the last two seconds are kept in a buffer; every half
second the buffer becomes one window. Model 1 takes the window's peak |ax − mean|
and walks two threshold tests. Model 2 computes the 8 features and walks
the bigger tree. Both verdicts go out over MQTT; model 2's becomes the icon.
The trees are trained on the laptop with scikit-learn from the recorded
data, then exported as plain thresholds (`models.json`) so the board needs
only numpy.

## Which model works better, and why

Tested honestly: train on one recording session, test on the other
(leave-one-session-out, 338 two-second windows).

| | Accuracy | Main confusion |
|---|---|---|
| Model 1 — peak `ax` only | **71%** | walking ↔ running |
| Model 2 — all six channels | **85%** | some running → walking |

**Model 2 wins.** One peak number is a single "how hard did the foot move"
reading: standing is obvious, but a brisk walk and an easy jog land on
overlapping peaks. Model 2 also sees *rotation* (the gyro) and *rhythm*
(step frequency): pedaling is a smooth, steady circle, running is fast
sharp impacts.

Lesson from the data: the first version of model 2 used per-axis features
and scored **99%** when tested on the same recording but **70%** on a new
one, because the strap sat slightly differently each time and the tree
had learned the board's tilt. Switching to orientation-free magnitudes
(and removing gravity from model 1's peak) fixed that: 85% / 71%.

## Reflection questions

*(Answer in your own words; `train.py` prints the numbers.)*
