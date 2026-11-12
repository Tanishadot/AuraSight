# Detector conversion parity (TFLite vs desktop ONNX)

- Model: `yolo11n_320_float32.tflite`, input [1, 320, 320, 3] float32, output [1, 84, 2100], boxes normalised: True
- Reference detections (score >= 0.40): 21, matched: 21 (recall 100.00%)
- Mean IoU of matches: 0.964
- Mean score difference (mobile - desktop): -0.018, max |diff| 0.176

Expected differences: the desktop pads to a multiple of 32 (non-square) while the phone model has a
fixed square 320x320 input; small resampling differences; TFLite vs ONNX Runtime kernels.
This report runs the exported TFLite file with the LiteRT Python runtime on the CI CPU; the Android
runtime on the phone is checked separately by the emulator integration test.

| Image | Ref | Matched | Missed | Extra (mobile only) |
|---|---|---|---|---|
| desk_laptop_cup_plant.jpg | 1 | 1 | - | - |
| empty_room.jpg | 0 | 0 | - | - |
| empty_room_windows.jpg | 0 | 0 | - | - |
| fp_door_as_refrigerator.jpg | 0 | 0 | - | - |
| fp_glass_transom_as_tv.jpg | 1 | 1 | - | - |
| fp_window_as_tv.jpg | 1 | 1 | - | - |
| multi_modern_living_room.jpg | 2 | 2 | - | - |
| multi_rural_living_room.jpg | 0 | 0 | - | - |
| person_desk_laptop_cup.jpg | 3 | 3 | - | - |
| person_living_room.jpg | 1 | 1 | - | - |
| table_objects.jpg | 5 | 5 | - | - |
| tv_control_room_monitors.jpg | 5 | 5 | - | person 0.52, laptop 0.48 |
| tv_living_room.jpg | 1 | 1 | - | - |
| tv_living_room_1998.jpg | 1 | 1 | - | - |
| window_door_living_room.jpg | 0 | 0 | - | - |
| window_large_historic.jpg | 0 | 0 | - | - |
