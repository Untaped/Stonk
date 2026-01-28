# stock_ai/config.py

class Config:
    def __init__(self):
        # Data settings
        self.data_path = 'stonk_download/training_data.csv'
        self.model_path = 'stock_ai/lightgbm_model_5pct.pkl'
        self.features_path = 'stock_ai/feature_names_5pct.pkl'
        
        # Model Target settings
        self.target_column = 'target_vol_adj_alpha'
        self.gain_threshold = 0.05  # 5% gain
        self.forward_periods = 5    # 5 days
        
        # Training settings
        self.test_size = 0.2
        self.random_state = 42