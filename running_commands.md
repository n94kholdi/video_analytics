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

## restricted area detection and track people movement in region
```
python -m app.analytics.cli \                                                                 
  data/human.mp4 \
  --camera-config configs/cameras/example_lobby.yaml \
  --enable-restricted-area \
  --output outputs/phase6_demo.mp4 \        
  --counts-csv outputs/phase6_demo_counts.csv \
  --events-jsonl outputs/phase6_demo_events.jsonl
```
## Heatmap computing

```
python -m app.analytics.cli \                                                                 
  data/human.mp4 \
  --camera-config configs/cameras/example_lobby.yaml \
  --output outputs/full_demo/annotated.mp4 \
  --counts-csv outputs/full_demo/counts.csv \
  --events-jsonl outputs/full_demo/events.jsonl \
  --heatmap-dir outputs/full_demo/heatmap \
  --enable-heatmap
```

## Queue analysis

```
python -m app.analytics.cli data/human.mp4 \                                                  
  --enable-queue \
  --queue-column-distance 0.04 \
  --queue-min-people 2 \
  --output outputs/vertical_queues.mp4 \
  --counts-csv outputs/vertical_queues.csv
```
## Queue speed
```
python -m app.analytics.cli data/human2.mp4 \                                                 
  --enable-queue \
  --queue-column-distance 0.04 \
  --queue-min-people 2 \
  --output outputs/vertical_queues_with_speed2.mp4 \
  --counts-csv outputs/vertical_queues_with_speed2.csv
```