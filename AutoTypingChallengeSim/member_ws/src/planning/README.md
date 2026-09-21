# planning

Types the launch key on the keyboard: aims the stylus at each key in turn and presses it once the arm has settled.
The panel's pose comes from the ArUco markers (`vision`); the keyboard's place on the panel is a stored offset.

```
planning/
  nodes/       the ROS nodes (entry points), each a thin wrapper over the code below
    key_sequence_node.py   `type_keys`  launch key    -> /planning/*_target, /arm/press
    kinematics_node.py     `kinematics` /planning/*_target -> /target_states  (joint targets for `arm-control`)
  geometry/    pure geometry, no ROS
    kinematics.py          the arm: forward and inverse kinematics, aiming the stylus
    panel.py               a flat panel (board, key grid) as a frame in the world; rays hitting it
    approach.py            where to put the head for a key, and when it is safe to press
    keyboard.py            the public key layout (INTERFACES.md 6.3) and where the key grid sits on the panel
  utils/       shared helpers
    ros.py                 joint states, TF frames, the launch key, message helpers, main()
    tasks.py               step-by-step tasks written as generators (sleep, hold, one at a time)
launch/        typing.launch.py
```

Data flow: `vision` gives the panel's pose; `type_keys` turns each key of the launch key into a head and aim target
using the stored keyboard offset (`keyboard_origin`, default in `geometry/keyboard.py`); `kinematics` turns those into
joint targets; `arm-control` drives the joints. Everything is in metres and radians.
