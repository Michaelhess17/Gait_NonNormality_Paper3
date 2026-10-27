# Leg indices (used to index into the 2-element parameter vectors)
const LEFT  = 1
const RIGHT = 2

# Hybrid discrete states (the "regime" reported in the output)
#   SS_LEFT       single support, left leg only on the ground
#   SS_RIGHT      single support, right leg only on the ground
#   DS_LEAD_LEFT  double support, LEFT leg is the leading (front) leg
#                 (just touched down); RIGHT leg is trailing and about
#                 to lift off
#   DS_LEAD_RIGHT double support, RIGHT leg is the leading (front) leg
#                 (just touched down); LEFT leg is trailing and about
#                 to lift off
const SS_LEFT       = 1
const SS_RIGHT       = 2
const DS_LEAD_LEFT   = 3
const DS_LEAD_RIGHT  = 4
