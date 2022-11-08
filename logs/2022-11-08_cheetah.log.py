config = {
    "evaluation": {
        "enabled": True, 
        "number_of_episodes_before_eval": 550, 
        "number_of_epsiodes_during_eval": 10, 
        "final_eval": {
            "number_of_steps": None, 
            "number_of_episodes": 10, 
        }, 
    }, 
    "verbose": False, 
    "number_of_processes": 10, 
    "number_of_malicious_processes": 0, 
    "logarithmic_scale_reference": [
        0.1, 
        0.08, 
        0.063, 
        0.05, 
        0.04, 
        0.032, 
        0.025, 
        0.02, 
        0.016, 
        0.012, 
        0.01, 
        0.008, 
        0.0063, 
        0.005, 
        0.004, 
        0.0032, 
        0.0025, 
        0.002, 
        0.0016, 
        0.0012, 
        0.001, 
        0.0008, 
        0.00063, 
        0.0005, 
        0.0004, 
        0.00032, 
        0.00025, 
        0.0002, 
        0.00016, 
        0.00012, 
        1e-05, 
        8e-06, 
        6.3e-06, 
        5e-06, 
        4e-06, 
        3.2e-06, 
        2.5e-06, 
        2e-06, 
        1.6e-06, 
        1.2e-06, 
    ], 
    "attack_method": "sign", 
    "use_frozen_random_seed": False, 
    "random_seeds": [0, ], 
    "value_trend_lookback_size": 10, 
    "early_stopping": {
        "lowerbound_for_max_recent_change": 0.1, 
        "min_number_of_episodes": 10, 
        "thresholds": {10: -700, }, 
    }, 
    "training": {"episode_count": 2000, }, 
    "env_config": {
        "env_name": "HalfCheetah-v2", 
        "learning_rate": 0.0002, 
        "beta": 2e-05, 
        "t_max": 5, 
        "activation": 1, 
        "hidden_size": 128, 
        "variance_scaling_factor": 100, 
        "permaban_threshold": 1000, 
    }, 
    "tuning": {
        "phase_1": {
            "categorical_options": {"activation": [0, 1, 2, ], }, 
            "sequential_options": {
                "t_max": [3, 5, 10, 20, 30, 50, ], 
                "hidden_size": [16, 32, 64, 128, ], 
            }, 
            "sequential_exponential_options": {
                "learning_rate": {
                    "base": [1, 2, 3, 4, ], 
                    "exponent": [-1, -2, -3, -4, -5, -6, -7, -8, ], 
                }, 
                "beta": {
                    "base": [1, 2, 3, 4, ], 
                    "exponent": [-1, -2, -3, -4, -5, -6, -7, -8, ], 
                }, 
            }, 
        }, 
        "phase_2": {
            "sequential_options": {
                "t_max": [3, 5, ], 
                "hidden_size": [64, 128, ], 
                "variance_scaling_factor": [0.01, 1, 100, 1000, ], 
                "permaban_threshold": [100, 500, 1000, 2000, 10000, ], 
            }, 
            "sequential_exponential_options": {
                "learning_rate": {"base": [2, 3, ], "exponent": [-4, -5, ], }, 
                "beta": {
                    "base": [1, 2, 3, 4, ], 
                    "exponent": [-4, -5, -6, -7, ], 
                }, 
            }, 
        }, 
    }, 
}
args = {
    "processes": 10, 
    "env": "HalfCheetah-v2", 
    "seed": 3646157750, 
    "outdir": "results", 
    "t_max": 5, 
    "beta": 2e-05, 
    "profile": False, 
    "steps": 2000, 
    "max_frames": (108000, ), 
    "lr": 0.0002, 
    "demo": False, 
    "load_pretrained": False, 
    "pretrained_type": "best", 
    "load": "", 
    "log_level": 20, 
    "render": False, 
    "monitor": False, 
    "permaban_threshold": 1000, 
    "malicious": 0, 
    "mal_type": "sign", 
    "rew_scale": 1.0, 
    "hidden_size": 128, 
    "activation": 1, 
}

{"total_number_of_episodes": 20, "number_of_timesteps": 20000, "per_episode_reward": -618.29, "episode_reward_trend_value": 0.0, "biggest_recent_change": NaN},
{"total_number_of_episodes": 30, "number_of_timesteps": 30000, "per_episode_reward": -610.53, "episode_reward_trend_value": 0.7762737349193685, "biggest_recent_change": NaN},
{"total_number_of_episodes": 40, "number_of_timesteps": 40000, "per_episode_reward": -599.11, "episode_reward_trend_value": 0.9588104896850893, "biggest_recent_change": NaN},
{"total_number_of_episodes": 50, "number_of_timesteps": 50000, "per_episode_reward": -592.98, "episode_reward_trend_value": 0.8434947828751509, "biggest_recent_change": NaN},
{"total_number_of_episodes": 60, "number_of_timesteps": 60000, "per_episode_reward": -595.91, "episode_reward_trend_value": 0.5593744099566151, "biggest_recent_change": NaN},
{"total_number_of_episodes": 70, "number_of_timesteps": 70000, "per_episode_reward": -589.36, "episode_reward_trend_value": 0.5785474529289762, "biggest_recent_change": NaN},
{"total_number_of_episodes": 80, "number_of_timesteps": 80000, "per_episode_reward": -597.35, "episode_reward_trend_value": 0.3489942188568231, "biggest_recent_change": NaN},
{"total_number_of_episodes": 90, "number_of_timesteps": 90000, "per_episode_reward": -587.75, "episode_reward_trend_value": 0.43625861332903765, "biggest_recent_change": NaN},
{"total_number_of_episodes": 100, "number_of_timesteps": 100000, "per_episode_reward": -588.06, "episode_reward_trend_value": 0.3778929174141382, "biggest_recent_change": NaN},
{"total_number_of_episodes": 110, "number_of_timesteps": 110000, "per_episode_reward": -590.45, "episode_reward_trend_value": 0.309324678868269, "biggest_recent_change": 11.413472444508102},
{"total_number_of_episodes": 120, "number_of_timesteps": 120000, "per_episode_reward": -585.32, "episode_reward_trend_value": 0.28004332507899915, "biggest_recent_change": 11.413472444508102},
{"total_number_of_episodes": 130, "number_of_timesteps": 130000, "per_episode_reward": -583.01, "episode_reward_trend_value": 0.1789686522262956, "biggest_recent_change": 9.598449801623246},
{"total_number_of_episodes": 140, "number_of_timesteps": 140000, "per_episode_reward": -577.94, "episode_reward_trend_value": 0.16712996062293922, "biggest_recent_change": 9.598449801623246},
{"total_number_of_episodes": 150, "number_of_timesteps": 150000, "per_episode_reward": -573.28, "episode_reward_trend_value": 0.25151268614257055, "biggest_recent_change": 9.598449801623246},
{"total_number_of_episodes": 160, "number_of_timesteps": 160000, "per_episode_reward": -570.39, "episode_reward_trend_value": 0.21073718217292403, "biggest_recent_change": 9.598449801623246},
{"total_number_of_episodes": 170, "number_of_timesteps": 170000, "per_episode_reward": -565.17, "episode_reward_trend_value": 0.35752164298371175, "biggest_recent_change": 9.598449801623246},
{"total_number_of_episodes": 180, "number_of_timesteps": 180000, "per_episode_reward": -563.81, "episode_reward_trend_value": 0.26597563525569967, "biggest_recent_change": 5.222881957931463},
{"total_number_of_episodes": 190, "number_of_timesteps": 190000, "per_episode_reward": -556.76, "episode_reward_trend_value": 0.3477559038381843, "biggest_recent_change": 7.053554632522037},
{"total_number_of_episodes": 200, "number_of_timesteps": 200000, "per_episode_reward": -554.8, "episode_reward_trend_value": 0.39606565503272073, "biggest_recent_change": 7.053554632522037},
{"total_number_of_episodes": 210, "number_of_timesteps": 210000, "per_episode_reward": -550.48, "episode_reward_trend_value": 0.38708530214708314, "biggest_recent_change": 7.053554632522037},
{"total_number_of_episodes": 220, "number_of_timesteps": 220000, "per_episode_reward": -543.42, "episode_reward_trend_value": 0.4398365489496541, "biggest_recent_change": 7.064364099996169},
{"total_number_of_episodes": 230, "number_of_timesteps": 230000, "per_episode_reward": -536.36, "episode_reward_trend_value": 0.46207224508015843, "biggest_recent_change": 7.064364099996169},
{"total_number_of_episodes": 240, "number_of_timesteps": 240000, "per_episode_reward": -533.74, "episode_reward_trend_value": 0.43928451068295293, "biggest_recent_change": 7.064364099996169},
{"total_number_of_episodes": 250, "number_of_timesteps": 250000, "per_episode_reward": -531.18, "episode_reward_trend_value": 0.4357406572805404, "biggest_recent_change": 7.064364099996169},
{"total_number_of_episodes": 260, "number_of_timesteps": 260000, "per_episode_reward": -526.98, "episode_reward_trend_value": 0.4243152632972005, "biggest_recent_change": 7.064364099996169},
{"total_number_of_episodes": 270, "number_of_timesteps": 270000, "per_episode_reward": -522.79, "episode_reward_trend_value": 0.4558184565564097, "biggest_recent_change": 7.064364099996169},
{"total_number_of_episodes": 280, "number_of_timesteps": 280000, "per_episode_reward": -517.47, "episode_reward_trend_value": 0.43656441086743347, "biggest_recent_change": 7.064364099996169},
{"total_number_of_episodes": 290, "number_of_timesteps": 290000, "per_episode_reward": -512.15, "episode_reward_trend_value": 0.47395357984513187, "biggest_recent_change": 7.064364099996169},
{"total_number_of_episodes": 300, "number_of_timesteps": 300000, "per_episode_reward": -503.14, "episode_reward_trend_value": 0.5260341642180638, "biggest_recent_change": 9.006436342015888},
{"total_number_of_episodes": 310, "number_of_timesteps": 310000, "per_episode_reward": -494.13, "episode_reward_trend_value": 0.5476127446849495, "biggest_recent_change": 9.006436342015888},
{"total_number_of_episodes": 320, "number_of_timesteps": 320000, "per_episode_reward": -486.6, "episode_reward_trend_value": 0.5528504506768223, "biggest_recent_change": 9.006436342015888},
{"total_number_of_episodes": 330, "number_of_timesteps": 330000, "per_episode_reward": -479.06, "episode_reward_trend_value": 0.6075401787461121, "biggest_recent_change": 9.006436342015888},
{"total_number_of_episodes": 340, "number_of_timesteps": 340000, "per_episode_reward": -475.06, "episode_reward_trend_value": 0.6235093184675191, "biggest_recent_change": 9.006436342015888},
{"total_number_of_episodes": 350, "number_of_timesteps": 350000, "per_episode_reward": -471.06, "episode_reward_trend_value": 0.621356875803016, "biggest_recent_change": 9.006436342015888},
{"total_number_of_episodes": 360, "number_of_timesteps": 360000, "per_episode_reward": -465.01, "episode_reward_trend_value": 0.6420384040300469, "biggest_recent_change": 9.006436342015888},
{"total_number_of_episodes": 370, "number_of_timesteps": 370000, "per_episode_reward": -458.95, "episode_reward_trend_value": 0.6502077764672638, "biggest_recent_change": 9.006436342015888},
{"total_number_of_episodes": 380, "number_of_timesteps": 380000, "per_episode_reward": -453.04, "episode_reward_trend_value": 0.6567383029474299, "biggest_recent_change": 9.006436342015888},
{"total_number_of_episodes": 390, "number_of_timesteps": 390000, "per_episode_reward": -447.13, "episode_reward_trend_value": 0.6223160980775788, "biggest_recent_change": 9.006436342015888},
{"total_number_of_episodes": 400, "number_of_timesteps": 400000, "per_episode_reward": -443.19, "episode_reward_trend_value": 0.5660451455901953, "biggest_recent_change": 7.535757639264602},
{"total_number_of_episodes": 410, "number_of_timesteps": 410000, "per_episode_reward": -439.25, "episode_reward_trend_value": 0.526115067577826, "biggest_recent_change": 7.535757639264489},
{"total_number_of_episodes": 420, "number_of_timesteps": 420000, "per_episode_reward": -431.65, "episode_reward_trend_value": 0.5268542359818051, "biggest_recent_change": 7.602282795622614},
{"total_number_of_episodes": 430, "number_of_timesteps": 430000, "per_episode_reward": -424.04, "episode_reward_trend_value": 0.5668698597151054, "biggest_recent_change": 7.602282795622614},
{"total_number_of_episodes": 440, "number_of_timesteps": 440000, "per_episode_reward": -419.49, "episode_reward_trend_value": 0.5729982297871142, "biggest_recent_change": 7.602282795622614},
{"total_number_of_episodes": 450, "number_of_timesteps": 450000, "per_episode_reward": -414.94, "episode_reward_trend_value": 0.556292628967588, "biggest_recent_change": 7.602282795622614},
{"total_number_of_episodes": 460, "number_of_timesteps": 460000, "per_episode_reward": -410.39, "episode_reward_trend_value": 0.5396065762212775, "biggest_recent_change": 7.602282795622614},
{"total_number_of_episodes": 470, "number_of_timesteps": 470000, "per_episode_reward": -405.83, "episode_reward_trend_value": 0.5245593694320153, "biggest_recent_change": 7.602282795622614},
{"total_number_of_episodes": 480, "number_of_timesteps": 480000, "per_episode_reward": -402.32, "episode_reward_trend_value": 0.49790766837431055, "biggest_recent_change": 7.602282795622614},
{"total_number_of_episodes": 490, "number_of_timesteps": 490000, "per_episode_reward": -397.43, "episode_reward_trend_value": 0.5085020957996935, "biggest_recent_change": 7.602282795622614},
{"total_number_of_episodes": 500, "number_of_timesteps": 500000, "per_episode_reward": -391.92, "episode_reward_trend_value": 0.525883319376671, "biggest_recent_change": 7.602282795622614},
{"total_number_of_episodes": 510, "number_of_timesteps": 510000, "per_episode_reward": -386.41, "episode_reward_trend_value": 0.5025952965373013, "biggest_recent_change": 7.602282795622557},
{"total_number_of_episodes": 520, "number_of_timesteps": 520000, "per_episode_reward": -381.16, "episode_reward_trend_value": 0.47650960413453225, "biggest_recent_change": 5.506360740079344},
{"total_number_of_episodes": 530, "number_of_timesteps": 530000, "per_episode_reward": -375.9, "episode_reward_trend_value": 0.4843111653930533, "biggest_recent_change": 5.506360740079344},
{"total_number_of_episodes": 540, "number_of_timesteps": 540000, "per_episode_reward": -369.83, "episode_reward_trend_value": 0.5012397216698123, "biggest_recent_change": 6.076000031014701},
{"total_number_of_episodes": 550, "number_of_timesteps": 550000, "per_episode_reward": -363.75, "episode_reward_trend_value": 0.5181487298733556, "biggest_recent_change": 6.076000031014701},
{"total_number_of_episodes": 560, "number_of_timesteps": 560000, "per_episode_reward": -352.49, "episode_reward_trend_value": 0.5926938090489633, "biggest_recent_change": 11.26324641850033},
{"total_number_of_episodes": 570, "number_of_timesteps": 570000, "per_episode_reward": -345.15, "episode_reward_trend_value": 0.6351932556059694, "biggest_recent_change": 11.26324641850033},
{"total_number_of_episodes": 580, "number_of_timesteps": 580000, "per_episode_reward": -340.06, "episode_reward_trend_value": 0.6373924769949542, "biggest_recent_change": 11.26324641850033},
{"total_number_of_episodes": 590, "number_of_timesteps": 590000, "per_episode_reward": -334.97, "episode_reward_trend_value": 0.6328049022323436, "biggest_recent_change": 11.26324641850033},
{"total_number_of_episodes": 600, "number_of_timesteps": 600000, "per_episode_reward": -330.86, "episode_reward_trend_value": 0.6171996212694284, "biggest_recent_change": 11.26324641850033},
{"total_number_of_episodes": 610, "number_of_timesteps": 610000, "per_episode_reward": -326.76, "episode_reward_trend_value": 0.604392009869913, "biggest_recent_change": 11.26324641850033},
{"total_number_of_episodes": 620, "number_of_timesteps": 620000, "per_episode_reward": -322.12, "episode_reward_trend_value": 0.5976449354549755, "biggest_recent_change": 11.26324641850033},
{"total_number_of_episodes": 630, "number_of_timesteps": 630000, "per_episode_reward": -317.47, "episode_reward_trend_value": 0.5817708660218001, "biggest_recent_change": 11.26324641850033},
{"total_number_of_episodes": 640, "number_of_timesteps": 640000, "per_episode_reward": -312.8, "episode_reward_trend_value": 0.5661373113106714, "biggest_recent_change": 11.26324641850033},
{"total_number_of_episodes": 650, "number_of_timesteps": 650000, "per_episode_reward": -308.13, "episode_reward_trend_value": 0.4928676856274807, "biggest_recent_change": 7.33473499866642},
{"total_number_of_episodes": 660, "number_of_timesteps": 660000, "per_episode_reward": -305.35, "episode_reward_trend_value": 0.44222075163025354, "biggest_recent_change": 5.093479011444458},
{"total_number_of_episodes": 670, "number_of_timesteps": 670000, "per_episode_reward": -302.58, "episode_reward_trend_value": 0.4164766619354926, "biggest_recent_change": 5.093479011444401},
{"total_number_of_episodes": 680, "number_of_timesteps": 680000, "per_episode_reward": -297.36, "episode_reward_trend_value": 0.41787169152358067, "biggest_recent_change": 5.2190316743723315},
{"total_number_of_episodes": 690, "number_of_timesteps": 690000, "per_episode_reward": -292.14, "episode_reward_trend_value": 0.4302844273119736, "biggest_recent_change": 5.2190316743723315},
{"total_number_of_episodes": 700, "number_of_timesteps": 700000, "per_episode_reward": -288.91, "episode_reward_trend_value": 0.42059281937968834, "biggest_recent_change": 5.2190316743723315},
{"total_number_of_episodes": 710, "number_of_timesteps": 710000, "per_episode_reward": -285.68, "episode_reward_trend_value": 0.404840674462826, "biggest_recent_change": 5.2190316743723315},
{"total_number_of_episodes": 720, "number_of_timesteps": 720000, "per_episode_reward": -283.97, "episode_reward_trend_value": 0.37217551639784574, "biggest_recent_change": 5.2190316743723315},
{"total_number_of_episodes": 730, "number_of_timesteps": 730000, "per_episode_reward": -282.27, "episode_reward_trend_value": 0.33926984361081874, "biggest_recent_change": 5.2190316743723315},
{"total_number_of_episodes": 740, "number_of_timesteps": 740000, "per_episode_reward": -279.43, "episode_reward_trend_value": 0.31894125755353697, "biggest_recent_change": 5.2190316743723315},
{"total_number_of_episodes": 750, "number_of_timesteps": 750000, "per_episode_reward": -276.59, "episode_reward_trend_value": 0.3196401066973339, "biggest_recent_change": 5.2190316743723315},
{"total_number_of_episodes": 760, "number_of_timesteps": 760000, "per_episode_reward": -274.71, "episode_reward_trend_value": 0.30968033899460023, "biggest_recent_change": 5.2190316743723315},
{"total_number_of_episodes": 770, "number_of_timesteps": 770000, "per_episode_reward": -272.83, "episode_reward_trend_value": 0.2725814520090163, "biggest_recent_change": 5.2190316743723315},
{"total_number_of_episodes": 780, "number_of_timesteps": 780000, "per_episode_reward": -271.74, "episode_reward_trend_value": 0.22661543503817633, "biggest_recent_change": 3.2296407395112965},
{"total_number_of_episodes": 790, "number_of_timesteps": 790000, "per_episode_reward": -270.66, "episode_reward_trend_value": 0.20275376178801516, "biggest_recent_change": 3.2296407395112965},
{"total_number_of_episodes": 800, "number_of_timesteps": 800000, "per_episode_reward": -267.6, "episode_reward_trend_value": 0.200906931517309, "biggest_recent_change": 3.0634260151477406},
{"total_number_of_episodes": 810, "number_of_timesteps": 810000, "per_episode_reward": -264.53, "episode_reward_trend_value": 0.21597311439472072, "biggest_recent_change": 3.0634260151477406},
{"total_number_of_episodes": 820, "number_of_timesteps": 820000, "per_episode_reward": -261.68, "episode_reward_trend_value": 0.2287256590422689, "biggest_recent_change": 3.0634260151477406},
{"total_number_of_episodes": 830, "number_of_timesteps": 830000, "per_episode_reward": -258.82, "episode_reward_trend_value": 0.22890111696007126, "biggest_recent_change": 3.0634260151477406},
{"total_number_of_episodes": 840, "number_of_timesteps": 840000, "per_episode_reward": -256.86, "episode_reward_trend_value": 0.21913762283504576, "biggest_recent_change": 3.0634260151477406},
{"total_number_of_episodes": 850, "number_of_timesteps": 850000, "per_episode_reward": -254.9, "episode_reward_trend_value": 0.22003274555654978, "biggest_recent_change": 3.0634260151477406},
{"total_number_of_episodes": 860, "number_of_timesteps": 860000, "per_episode_reward": -252.93, "episode_reward_trend_value": 0.2210424742578681, "biggest_recent_change": 3.0634260151477406},
{"total_number_of_episodes": 870, "number_of_timesteps": 870000, "per_episode_reward": -250.96, "episode_reward_trend_value": 0.2309193329444434, "biggest_recent_change": 3.0634260151477406},
{"total_number_of_episodes": 880, "number_of_timesteps": 880000, "per_episode_reward": -247.29, "episode_reward_trend_value": 0.25963430168839857, "biggest_recent_change": 3.666437333952757},
{"total_number_of_episodes": 890, "number_of_timesteps": 890000, "per_episode_reward": -245.62, "episode_reward_trend_value": 0.24417365409275454, "biggest_recent_change": 3.666437333952757},
{"total_number_of_episodes": 900, "number_of_timesteps": 900000, "per_episode_reward": -242.47, "episode_reward_trend_value": 0.24519533961997633, "biggest_recent_change": 3.666437333952757},
{"total_number_of_episodes": 910, "number_of_timesteps": 910000, "per_episode_reward": -239.31, "episode_reward_trend_value": 0.2485306633770632, "biggest_recent_change": 3.666437333952757},
{"total_number_of_episodes": 920, "number_of_timesteps": 920000, "per_episode_reward": -237.96, "episode_reward_trend_value": 0.23181416487486278, "biggest_recent_change": 3.666437333952757},
{"total_number_of_episodes": 930, "number_of_timesteps": 930000, "per_episode_reward": -236.61, "episode_reward_trend_value": 0.2250366184154915, "biggest_recent_change": 3.666437333952757},
{"total_number_of_episodes": 940, "number_of_timesteps": 940000, "per_episode_reward": -235.44, "episode_reward_trend_value": 0.21622158355993468, "biggest_recent_change": 3.666437333952757},
{"total_number_of_episodes": 950, "number_of_timesteps": 950000, "per_episode_reward": -234.87, "episode_reward_trend_value": 0.20073911447893206, "biggest_recent_change": 3.666437333952757},
{"total_number_of_episodes": 960, "number_of_timesteps": 960000, "per_episode_reward": -233.16, "episode_reward_trend_value": 0.1978009572778714, "biggest_recent_change": 3.666437333952757},
{"total_number_of_episodes": 970, "number_of_timesteps": 970000, "per_episode_reward": -231.45, "episode_reward_trend_value": 0.1760246900194302, "biggest_recent_change": 3.155377712597783},
{"total_number_of_episodes": 980, "number_of_timesteps": 980000, "per_episode_reward": -229.47, "episode_reward_trend_value": 0.179518816047817, "biggest_recent_change": 3.155377712597783},
{"total_number_of_episodes": 990, "number_of_timesteps": 990000, "per_episode_reward": -227.48, "episode_reward_trend_value": 0.16653060895333865, "biggest_recent_change": 3.155377712597783},
{"total_number_of_episodes": 1000, "number_of_timesteps": 1000000, "per_episode_reward": -225.83, "episode_reward_trend_value": 0.14984170386170267, "biggest_recent_change": 1.9864390740946476},
{"total_number_of_episodes": 1010, "number_of_timesteps": 1010000, "per_episode_reward": -224.83, "episode_reward_trend_value": 0.1459515548325388, "biggest_recent_change": 1.9864390740946476},
{"total_number_of_episodes": 1020, "number_of_timesteps": 1020000, "per_episode_reward": -222.74, "episode_reward_trend_value": 0.1541325115239674, "biggest_recent_change": 2.086999811490557},
{"total_number_of_episodes": 1030, "number_of_timesteps": 1030000, "per_episode_reward": -220.65, "episode_reward_trend_value": 0.16435095661158117, "biggest_recent_change": 2.086999811490557},
{"total_number_of_episodes": 1040, "number_of_timesteps": 1040000, "per_episode_reward": -219.28, "episode_reward_trend_value": 0.17322418948110618, "biggest_recent_change": 2.086999811490557},
{"total_number_of_episodes": 1050, "number_of_timesteps": 1050000, "per_episode_reward": -217.9, "episode_reward_trend_value": 0.16955311047068827, "biggest_recent_change": 2.086999811490557},
{"total_number_of_episodes": 1060, "number_of_timesteps": 1060000, "per_episode_reward": -216.31, "episode_reward_trend_value": 0.16821438334083832, "biggest_recent_change": 2.086999811490557},
{"total_number_of_episodes": 1070, "number_of_timesteps": 1070000, "per_episode_reward": -214.73, "episode_reward_trend_value": 0.16376603628430392, "biggest_recent_change": 2.086999811490557},
{"total_number_of_episodes": 1080, "number_of_timesteps": 1080000, "per_episode_reward": -213.67, "episode_reward_trend_value": 0.15341551841463633, "biggest_recent_change": 2.086999811490557},
{"total_number_of_episodes": 1090, "number_of_timesteps": 1090000, "per_episode_reward": -212.62, "episode_reward_trend_value": 0.14676569854212573, "biggest_recent_change": 2.086999811490557},
{"total_number_of_episodes": 1100, "number_of_timesteps": 1100000, "per_episode_reward": -210.81, "episode_reward_trend_value": 0.1557150668868041, "biggest_recent_change": 2.086999811490557},
{"total_number_of_episodes": 1110, "number_of_timesteps": 1110000, "per_episode_reward": -207.64, "episode_reward_trend_value": 0.1677430926441032, "biggest_recent_change": 3.1695221296474756},
{"total_number_of_episodes": 1120, "number_of_timesteps": 1120000, "per_episode_reward": -205.82, "episode_reward_trend_value": 0.16481975892826028, "biggest_recent_change": 3.1695221296474756},
{"total_number_of_episodes": 1130, "number_of_timesteps": 1130000, "per_episode_reward": -203.99, "episode_reward_trend_value": 0.1697944656761403, "biggest_recent_change": 3.1695221296474756},
{"total_number_of_episodes": 1140, "number_of_timesteps": 1140000, "per_episode_reward": -202.54, "episode_reward_trend_value": 0.17062244503768323, "biggest_recent_change": 3.1695221296474756},
{"total_number_of_episodes": 1150, "number_of_timesteps": 1150000, "per_episode_reward": -201.01, "episode_reward_trend_value": 0.170070783719528, "biggest_recent_change": 3.1695221296474756},
{"total_number_of_episodes": 1160, "number_of_timesteps": 1160000, "per_episode_reward": -199.97, "episode_reward_trend_value": 0.16401180451604616, "biggest_recent_change": 3.1695221296474756},
{"total_number_of_episodes": 1170, "number_of_timesteps": 1170000, "per_episode_reward": -199.15, "episode_reward_trend_value": 0.16140012647553179, "biggest_recent_change": 3.1695221296474756},
{"total_number_of_episodes": 1180, "number_of_timesteps": 1180000, "per_episode_reward": -198.12, "episode_reward_trend_value": 0.16103341005899224, "biggest_recent_change": 3.1695221296474756},
{"total_number_of_episodes": 1190, "number_of_timesteps": 1190000, "per_episode_reward": -197.1, "episode_reward_trend_value": 0.1523205716220774, "biggest_recent_change": 3.1695221296474756},
{"total_number_of_episodes": 1200, "number_of_timesteps": 1200000, "per_episode_reward": -196.05, "episode_reward_trend_value": 0.12879494551486376, "biggest_recent_change": 1.8238997770646392},
{"total_number_of_episodes": 1210, "number_of_timesteps": 1210000, "per_episode_reward": -195.0, "episode_reward_trend_value": 0.12022067888079277, "biggest_recent_change": 1.8238997770646392},
{"total_number_of_episodes": 1220, "number_of_timesteps": 1220000, "per_episode_reward": -194.21, "episode_reward_trend_value": 0.10868884837981656, "biggest_recent_change": 1.5364383203725822},
{"total_number_of_episodes": 1230, "number_of_timesteps": 1230000, "per_episode_reward": -193.69, "episode_reward_trend_value": 0.09839213362845606, "biggest_recent_change": 1.5364383203725822},
{"total_number_of_episodes": 1240, "number_of_timesteps": 1240000, "per_episode_reward": -192.84, "episode_reward_trend_value": 0.0907553583632588, "biggest_recent_change": 1.0522157799982494},
{"total_number_of_episodes": 1250, "number_of_timesteps": 1250000, "per_episode_reward": -191.99, "episode_reward_trend_value": 0.08862590098338816, "biggest_recent_change": 1.0522157799982494},
{"total_number_of_episodes": 1260, "number_of_timesteps": 1260000, "per_episode_reward": -191.51, "episode_reward_trend_value": 0.08486134947922482, "biggest_recent_change": 1.0522157799982494},
{"total_number_of_episodes": 1270, "number_of_timesteps": 1270000, "per_episode_reward": -191.16, "episode_reward_trend_value": 0.07742667483175611, "biggest_recent_change": 1.0522157799982494},
{"total_number_of_episodes": 1280, "number_of_timesteps": 1280000, "per_episode_reward": -190.65, "episode_reward_trend_value": 0.07170155058598512, "biggest_recent_change": 1.0522157799982494},
{"total_number_of_episodes": 1290, "number_of_timesteps": 1290000, "per_episode_reward": -189.87, "episode_reward_trend_value": 0.06861628755226769, "biggest_recent_change": 1.0522157799981926},
{"total_number_of_episodes": 1300, "number_of_timesteps": 1300000, "per_episode_reward": -190.03, "episode_reward_trend_value": 0.05524665864350785, "biggest_recent_change": 0.8491285465048293},
{"total_number_of_episodes": 1310, "number_of_timesteps": 1310000, "per_episode_reward": -188.99, "episode_reward_trend_value": 0.05796948817513416, "biggest_recent_change": 1.0310896898231476},
{"total_number_of_episodes": 1320, "number_of_timesteps": 1320000, "per_episode_reward": -187.33, "episode_reward_trend_value": 0.07068222536764779, "biggest_recent_change": 1.6681363319980846},
{"total_number_of_episodes": 1330, "number_of_timesteps": 1330000, "per_episode_reward": -185.66, "episode_reward_trend_value": 0.07978231187312777, "biggest_recent_change": 1.6681363319980846},
{"total_number_of_episodes": 1340, "number_of_timesteps": 1340000, "per_episode_reward": -186.6, "episode_reward_trend_value": 0.059898650262800855, "biggest_recent_change": 1.6681363319980846},
{"total_number_of_episodes": 1350, "number_of_timesteps": 1350000, "per_episode_reward": -185.28, "episode_reward_trend_value": 0.06918051980931574, "biggest_recent_change": 1.6681363319980846},
{"total_number_of_episodes": 1360, "number_of_timesteps": 1360000, "per_episode_reward": -184.28, "episode_reward_trend_value": 0.07638995663125393, "biggest_recent_change": 1.6681363319980846},
{"total_number_of_episodes": 1370, "number_of_timesteps": 1370000, "per_episode_reward": -183.28, "episode_reward_trend_value": 0.08188984305149537, "biggest_recent_change": 1.6681363319980846},
{"total_number_of_episodes": 1380, "number_of_timesteps": 1380000, "per_episode_reward": -182.07, "episode_reward_trend_value": 0.08674471744037299, "biggest_recent_change": 1.6681363319980846},
{"total_number_of_episodes": 1390, "number_of_timesteps": 1390000, "per_episode_reward": -180.86, "episode_reward_trend_value": 0.10188395770429363, "biggest_recent_change": 1.6681363319980846},
{"total_number_of_episodes": 1400, "number_of_timesteps": 1400000, "per_episode_reward": -180.47, "episode_reward_trend_value": 0.09473394004219339, "biggest_recent_change": 1.6681363319980846},
{"total_number_of_episodes": 1410, "number_of_timesteps": 1410000, "per_episode_reward": -180.08, "episode_reward_trend_value": 0.08050562635592781, "biggest_recent_change": 1.6681363319980278},
{"total_number_of_episodes": 1420, "number_of_timesteps": 1420000, "per_episode_reward": -179.22, "episode_reward_trend_value": 0.07148808744258461, "biggest_recent_change": 1.3164000659899102},
{"total_number_of_episodes": 1430, "number_of_timesteps": 1430000, "per_episode_reward": -178.37, "episode_reward_trend_value": 0.09145429664504925, "biggest_recent_change": 1.3164000659899102},
{"total_number_of_episodes": 1440, "number_of_timesteps": 1440000, "per_episode_reward": -177.7, "episode_reward_trend_value": 0.08429924844323326, "biggest_recent_change": 1.211480801962665},
{"total_number_of_episodes": 1450, "number_of_timesteps": 1450000, "per_episode_reward": -177.09, "episode_reward_trend_value": 0.07984925351825609, "biggest_recent_change": 1.211480801962665},
{"total_number_of_episodes": 1460, "number_of_timesteps": 1460000, "per_episode_reward": -175.7, "episode_reward_trend_value": 0.0842651068395652, "biggest_recent_change": 1.3990433829561084},
{"total_number_of_episodes": 1470, "number_of_timesteps": 1470000, "per_episode_reward": -174.3, "episode_reward_trend_value": 0.08634913551727044, "biggest_recent_change": 1.3990433829561368},
{"total_number_of_episodes": 1480, "number_of_timesteps": 1480000, "per_episode_reward": -172.75, "episode_reward_trend_value": 0.09006305091462188, "biggest_recent_change": 1.545733187724295},
{"total_number_of_episodes": 1490, "number_of_timesteps": 1490000, "per_episode_reward": -171.2, "episode_reward_trend_value": 0.10293132966451235, "biggest_recent_change": 1.545733187724295},
{"total_number_of_episodes": 1500, "number_of_timesteps": 1500000, "per_episode_reward": -170.32, "episode_reward_trend_value": 0.10847504862788133, "biggest_recent_change": 1.545733187724295},
{"total_number_of_episodes": 1510, "number_of_timesteps": 1510000, "per_episode_reward": -169.38, "episode_reward_trend_value": 0.10936372089327462, "biggest_recent_change": 1.545733187724295},
{"total_number_of_episodes": 1520, "number_of_timesteps": 1520000, "per_episode_reward": -168.3, "episode_reward_trend_value": 0.1118263314696675, "biggest_recent_change": 1.545733187724295},
{"total_number_of_episodes": 1530, "number_of_timesteps": 1530000, "per_episode_reward": -167.24, "episode_reward_trend_value": 0.11620344600280878, "biggest_recent_change": 1.545733187724295},
{"total_number_of_episodes": 1540, "number_of_timesteps": 1540000, "per_episode_reward": -166.23, "episode_reward_trend_value": 0.12067111433939255, "biggest_recent_change": 1.545733187724295},
{"total_number_of_episodes": 1550, "number_of_timesteps": 1550000, "per_episode_reward": -165.23, "episode_reward_trend_value": 0.11627293442969, "biggest_recent_change": 1.545733187724295},
{"total_number_of_episodes": 1560, "number_of_timesteps": 1560000, "per_episode_reward": -164.27, "episode_reward_trend_value": 0.1114546479586197, "biggest_recent_change": 1.545733187724295},
{"total_number_of_episodes": 1570, "number_of_timesteps": 1570000, "per_episode_reward": -163.3, "episode_reward_trend_value": 0.10500647476790322, "biggest_recent_change": 1.5457331877242666},
{"total_number_of_episodes": 1580, "number_of_timesteps": 1580000, "per_episode_reward": -162.18, "episode_reward_trend_value": 0.10031909422222673, "biggest_recent_change": 1.1238689386133842},
{"total_number_of_episodes": 1590, "number_of_timesteps": 1590000, "per_episode_reward": -161.05, "episode_reward_trend_value": 0.1029562734630711, "biggest_recent_change": 1.1238689386133842},
{"total_number_of_episodes": 1600, "number_of_timesteps": 1600000, "per_episode_reward": -159.65, "episode_reward_trend_value": 0.10814084064999305, "biggest_recent_change": 1.4031493805055106},
{"total_number_of_episodes": 1610, "number_of_timesteps": 1610000, "per_episode_reward": -158.25, "episode_reward_trend_value": 0.11175146952591446, "biggest_recent_change": 1.4031493805055106},
{"total_number_of_episodes": 1620, "number_of_timesteps": 1620000, "per_episode_reward": -157.66, "episode_reward_trend_value": 0.10643577135232743, "biggest_recent_change": 1.4031493805055106},
{"total_number_of_episodes": 1630, "number_of_timesteps": 1630000, "per_episode_reward": -157.07, "episode_reward_trend_value": 0.10182206034236571, "biggest_recent_change": 1.4031493805055106},
{"total_number_of_episodes": 1640, "number_of_timesteps": 1640000, "per_episode_reward": -156.2, "episode_reward_trend_value": 0.10037414156491561, "biggest_recent_change": 1.4031493805055106},
{"total_number_of_episodes": 1650, "number_of_timesteps": 1650000, "per_episode_reward": -155.32, "episode_reward_trend_value": 0.09934632934883295, "biggest_recent_change": 1.4031493805055106},
{"total_number_of_episodes": 1660, "number_of_timesteps": 1660000, "per_episode_reward": -152.3, "episode_reward_trend_value": 0.12222882409813626, "biggest_recent_change": 3.024822127997112},
{"total_number_of_episodes": 1670, "number_of_timesteps": 1670000, "per_episode_reward": -150.15, "episode_reward_trend_value": 0.13364579588977296, "biggest_recent_change": 3.024822127997112},
{"total_number_of_episodes": 1680, "number_of_timesteps": 1680000, "per_episode_reward": -149.06, "episode_reward_trend_value": 0.13323549151243191, "biggest_recent_change": 3.024822127997112},
{"total_number_of_episodes": 1690, "number_of_timesteps": 1690000, "per_episode_reward": -147.97, "episode_reward_trend_value": 0.12972207111406692, "biggest_recent_change": 3.024822127997112},
{"total_number_of_episodes": 1700, "number_of_timesteps": 1700000, "per_episode_reward": -147.15, "episode_reward_trend_value": 0.12330538768034609, "biggest_recent_change": 3.024822127997112},




Process Process-7:
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 599, in run_func
    f()
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 568, in f
    train_loop(
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 159, in train_loop
    all_updated_barrier.wait()  # Wait for all agents to complete rollout, then run when_all_processes_are_updated()
  File "/usr/lib/python3.8/threading.py", line 619, in wait
    self._wait(timeout)
  File "/usr/lib/python3.8/threading.py", line 659, in _wait
    raise BrokenBarrierError
threading.BrokenBarrierError




Process Process-9:
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 599, in run_func
    f()
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 568, in f
    train_loop(
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 159, in train_loop
    all_updated_barrier.wait()  # Wait for all agents to complete rollout, then run when_all_processes_are_updated()
  File "/usr/lib/python3.8/threading.py", line 619, in wait
    self._wait(timeout)
  File "/usr/lib/python3.8/threading.py", line 659, in _wait
    raise BrokenBarrierError
threading.BrokenBarrierError




Process Process-10:
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 599, in run_func
    f()
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 568, in f
    train_loop(
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 159, in train_loop
    all_updated_barrier.wait()  # Wait for all agents to complete rollout, then run when_all_processes_are_updated()
  File "/usr/lib/python3.8/threading.py", line 610, in wait
    self._enter() # Block while the barrier drains.
  File "/usr/lib/python3.8/threading.py", line 634, in _enter
    raise BrokenBarrierError
threading.BrokenBarrierError




Process Process-6:
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 599, in run_func
    f()
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 568, in f
    train_loop(
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 159, in train_loop
    all_updated_barrier.wait()  # Wait for all agents to complete rollout, then run when_all_processes_are_updated()
  File "/usr/lib/python3.8/threading.py", line 610, in wait
    self._enter() # Block while the barrier drains.
  File "/usr/lib/python3.8/threading.py", line 634, in _enter
    raise BrokenBarrierError
threading.BrokenBarrierError




Process Process-1:
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 599, in run_func
    f()
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 568, in f
    train_loop(
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 159, in train_loop
    all_updated_barrier.wait()  # Wait for all agents to complete rollout, then run when_all_processes_are_updated()
  File "/usr/lib/python3.8/threading.py", line 619, in wait
    self._wait(timeout)
  File "/usr/lib/python3.8/threading.py", line 659, in _wait
    raise BrokenBarrierError
threading.BrokenBarrierError




Process Process-3:
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 599, in run_func
    f()
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 568, in f
    train_loop(
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 159, in train_loop
    all_updated_barrier.wait()  # Wait for all agents to complete rollout, then run when_all_processes_are_updated()
  File "/usr/lib/python3.8/threading.py", line 619, in wait
    self._wait(timeout)
  File "/usr/lib/python3.8/threading.py", line 659, in _wait
    raise BrokenBarrierError
threading.BrokenBarrierError




Process Process-8:
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 599, in run_func
    f()
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 568, in f
    train_loop(
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 159, in train_loop
    all_updated_barrier.wait()  # Wait for all agents to complete rollout, then run when_all_processes_are_updated()
  File "/usr/lib/python3.8/threading.py", line 610, in wait
    self._enter() # Block while the barrier drains.
  File "/usr/lib/python3.8/threading.py", line 634, in _enter
    raise BrokenBarrierError
threading.BrokenBarrierError
{"total_number_of_episodes": 1710, "number_of_timesteps": 1710000, "per_episode_reward": -146.32, "episode_reward_trend_value": 0.12594621736128128, "biggest_recent_change": 3.024822127997112},
{"total_number_of_episodes": 1720, "number_of_timesteps": 1720000, "per_episode_reward": -144.9, "episode_reward_trend_value": 0.13517061228472377, "biggest_recent_change": 3.024822127997112},
{"total_number_of_episodes": 1730, "number_of_timesteps": 1730000, "per_episode_reward": -143.49, "episode_reward_trend_value": 0.14122921497565433, "biggest_recent_change": 3.024822127997112},
{"total_number_of_episodes": 1740, "number_of_timesteps": 1740000, "per_episode_reward": -141.73, "episode_reward_trend_value": 0.15105488949489257, "biggest_recent_change": 3.024822127997112},
{"total_number_of_episodes": 1750, "number_of_timesteps": 1750000, "per_episode_reward": -139.97, "episode_reward_trend_value": 0.13697025704874383, "biggest_recent_change": 2.151396399860687},
{"total_number_of_episodes": 1760, "number_of_timesteps": 1760000, "per_episode_reward": -138.89, "episode_reward_trend_value": 0.12512951553739968, "biggest_recent_change": 1.7572052078438105},
{"total_number_of_episodes": 1770, "number_of_timesteps": 1770000, "per_episode_reward": -137.8, "episode_reward_trend_value": 0.12511605019503297, "biggest_recent_change": 1.7572052078438105},
{"total_number_of_episodes": 1780, "number_of_timesteps": 1780000, "per_episode_reward": -136.54, "episode_reward_trend_value": 0.12704932026886126, "biggest_recent_change": 1.7572052078438105},
{"total_number_of_episodes": 1790, "number_of_timesteps": 1790000, "per_episode_reward": -135.28, "episode_reward_trend_value": 0.1318858533780463, "biggest_recent_change": 1.7572052078438105},
{"total_number_of_episodes": 1800, "number_of_timesteps": 1800000, "per_episode_reward": -133.03, "episode_reward_trend_value": 0.14769800532367455, "biggest_recent_change": 2.2487415465770653},
{"total_number_of_episodes": 1810, "number_of_timesteps": 1810000, "per_episode_reward": -130.78, "episode_reward_trend_value": 0.15692659202679618, "biggest_recent_change": 2.2487415465770937},
{"total_number_of_episodes": 1820, "number_of_timesteps": 1820000, "per_episode_reward": -129.68, "episode_reward_trend_value": 0.15335895267412586, "biggest_recent_change": 2.2487415465770937},
{"total_number_of_episodes": 1830, "number_of_timesteps": 1830000, "per_episode_reward": -128.59, "episode_reward_trend_value": 0.14602424149314794, "biggest_recent_change": 2.2487415465770937},
{"total_number_of_episodes": 1840, "number_of_timesteps": 1840000, "per_episode_reward": -127.1, "episode_reward_trend_value": 0.14303506344742312, "biggest_recent_change": 2.2487415465770937},
{"total_number_of_episodes": 1850, "number_of_timesteps": 1850000, "per_episode_reward": -125.61, "episode_reward_trend_value": 0.14750672477952062, "biggest_recent_change": 2.2487415465770937},
{"total_number_of_episodes": 1860, "number_of_timesteps": 1860000, "per_episode_reward": -124.05, "episode_reward_trend_value": 0.15280539593389966, "biggest_recent_change": 2.2487415465770937},
{"total_number_of_episodes": 1870, "number_of_timesteps": 1870000, "per_episode_reward": -122.49, "episode_reward_trend_value": 0.15615733167208343, "biggest_recent_change": 2.2487415465770937},
{"total_number_of_episodes": 1880, "number_of_timesteps": 1880000, "per_episode_reward": -121.53, "episode_reward_trend_value": 0.15279629495710323, "biggest_recent_change": 2.2487415465770937},
{"total_number_of_episodes": 1890, "number_of_timesteps": 1890000, "per_episode_reward": -120.57, "episode_reward_trend_value": 0.1384596394056803, "biggest_recent_change": 2.2487415465770937},
{"total_number_of_episodes": 1900, "number_of_timesteps": 1900000, "per_episode_reward": -119.1, "episode_reward_trend_value": 0.12981999161423627, "biggest_recent_change": 1.5626100677338002},
{"total_number_of_episodes": 1910, "number_of_timesteps": 1910000, "per_episode_reward": -117.63, "episode_reward_trend_value": 0.13397656987858444, "biggest_recent_change": 1.5626100677338002},
{"total_number_of_episodes": 1920, "number_of_timesteps": 1920000, "per_episode_reward": -116.95, "episode_reward_trend_value": 0.1293158617916316, "biggest_recent_change": 1.5626100677338002},
{"total_number_of_episodes": 1930, "number_of_timesteps": 1930000, "per_episode_reward": -116.17, "episode_reward_trend_value": 0.12138930611709593, "biggest_recent_change": 1.5626100677338002},
{"total_number_of_episodes": 1940, "number_of_timesteps": 1940000, "per_episode_reward": -114.95, "episode_reward_trend_value": 0.11849466288257038, "biggest_recent_change": 1.5626100677338002},
{"total_number_of_episodes": 1950, "number_of_timesteps": 1950000, "per_episode_reward": -113.72, "episode_reward_trend_value": 0.1147730098257638, "biggest_recent_change": 1.5626100677337433},
{"total_number_of_episodes": 1960, "number_of_timesteps": 1960000, "per_episode_reward": -112.48, "episode_reward_trend_value": 0.11118088863702427, "biggest_recent_change": 1.4711732453471313},
{"total_number_of_episodes": 1970, "number_of_timesteps": 1970000, "per_episode_reward": -111.24, "episode_reward_trend_value": 0.11430173990144869, "biggest_recent_change": 1.4711732453471313},
{"total_number_of_episodes": 1980, "number_of_timesteps": 1980000, "per_episode_reward": -110.42, "episode_reward_trend_value": 0.11275922355745346, "biggest_recent_change": 1.4711732453471313},
{"total_number_of_episodes": 1990, "number_of_timesteps": 1990000, "per_episode_reward": -109.6, "episode_reward_trend_value": 0.10551969945347883, "biggest_recent_change": 1.4711732453471313},




Process Process-2:
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 599, in run_func
    f()
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 568, in f
    train_loop(
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 159, in train_loop
    all_updated_barrier.wait()  # Wait for all agents to complete rollout, then run when_all_processes_are_updated()
  File "/usr/lib/python3.8/threading.py", line 610, in wait
    self._enter() # Block while the barrier drains.
  File "/usr/lib/python3.8/threading.py", line 634, in _enter
    raise BrokenBarrierError
threading.BrokenBarrierError




Process Process-4:
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 599, in run_func
    f()
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 568, in f
    train_loop(
  File "/home/jeffhykin/repos/bizav/pfrl/experiments/train_agent_async.py", line 159, in train_loop
    all_updated_barrier.wait()  # Wait for all agents to complete rollout, then run when_all_processes_are_updated()
  File "/usr/lib/python3.8/threading.py", line 610, in wait
    self._enter() # Block while the barrier drains.
  File "/usr/lib/python3.8/threading.py", line 634, in _enter
    raise BrokenBarrierError
threading.BrokenBarrierError
final_eval: {'number_of_steps': None, 'number_of_episodes': 10, 'mean': 372.63949058143436, 'median': 376.1350800766902, 'stdev': 33.49703000257739}
