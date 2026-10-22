# Day 8 — Gait Wearable (UNO Q + MPU-6050)

*ME193 — Chris Liang*

An ankle-worn UNO Q with an MPU-6050 six-axis IMU that tells **standing,
walking, running and biking** apart, with two models that use no neural nets:

- **Model 1** looks at one number: the **peak acceleration along one axis**
  (vertical, `az`) in a 2-second window, and splits it with two or three
  learned thresholds (a depth-2 decision tree).
- **Model 2** uses **all six IMU outputs**: 29 features per window (mean, std,
  min, max of each axis, the spread and dominant frequency of the acceleration
  magnitude, gyro energy) into a depth-4 decision tree.

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
second the buffer becomes one window. Model 1 takes the window's peak |az|
and walks two threshold tests. Model 2 computes the 29 features and walks
the bigger tree. Both verdicts go out over MQTT; model 2's becomes the icon.
The trees are trained on the laptop with scikit-learn from the recorded
data, then exported as plain thresholds (`models.json`) so the board needs
only numpy.

## Which model works better, and why

*(Fill in from `train.py`'s numbers after recording.)* Expect model 1 to
separate standing / walking / running well — those differ mostly in how hard
the foot hits the ground — and to confuse **biking** with walking, because
pedaling has gentle vertical acceleration but a lot of rotation. Model 2 sees
the gyro and the frequency, which is exactly what tells biking apart.

## Reflection questions

*(Answer in your own words; `train.py` prints the numbers.)*
