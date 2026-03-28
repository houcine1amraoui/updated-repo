ACTOR1_END    = "2022-11-07"
ACTOR2_START  = "2022-11-08"
ACTOR2_END    = "2022-11-10"
ACTOR1_RETURN = "2022-11-11"
RESAMPLE_FREQ = "3s"

window_slide          = 60
in_dim                = window_slide - 1
correlation_threshold = 0.5
nbr_sensors           = 94
NUM_SENSORS           = nbr_sensors
WINDOW                = window_slide

MTAD_BATCH    = 256
MTAD_EPOCHS   = 100
MTAD_LR       = 0.001
MTAD_PATIENCE = 15
ALPHA         = 0.5   # 50% forecast / 50% reconstruction


GDN_BATCH    = 256
GDN_EPOCHS   = 100
GDN_LR       = 0.001
GDN_PATIENCE = 15
