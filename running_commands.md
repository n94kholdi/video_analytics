# Demo

## human detection

```
python3 -m app.detection.cli \                                                                
  data/human.mp4 \
  --output outputs/human_detected-video.mp4 \
  --confidence 0.45 \
  --iou 0.70
```
```
python3 -m app.detection.cli \                                                                
  data/human.jpg \
  --output outputs/manual_test.jpg \         
  --confidence 0.40 \
  --iou 0.70
```

## Tracking
```
python3 -m app.tracking.cli \                                                                 
  data/human.mp4 \
  --output outputs/tracked_output.mp4
```
```
python3 -m app.tracking.cli \                                                                 
  data/human.mp4 \
  --output outputs/tracked_output_no_trajector.mp4 \
  --no-trajectories
```
## camera configuarion
```
python3 scripts/configure_camera.py data/human.jpg \                                          
  --camera-id test_camera \
  --name "Test Camera" \
  --source input.mp4 \
  --output outputs/test_camera.yaml
```

## Human counting
```
python3 -m app.analytics.cli data/human.mp4  \                                                
  --output outputs/human_counting.mp4 \
  --counts-csv outputs/human_counting.csv \
  --no-trajectories
```