
# for current architecture, UNet can have as many topology, however, UNet3+ is fixed to 5


class Config_Unet:
    
    class DATASET:
        MODE = 'fusion'

        SENTINEL1_BANDS = [0, 1]  # VV and VH bands  
        SENTINEL2_BANDS = [2, 3, 4, 5, 6, 7]  # 6 S2 bands 
    
    class MODEL:
        IN_CHANNELS = 8  # 2 S1 bands + 6 S2 bands  
        # IN_CHANNELS = 15
        OUT_CHANNELS = 2  # Binary classification (flood/no-flood)
        TOPOLOGY = [16 ,32, 64, 128, 256, 512, 1024]
 


class Config_Unet3P:
    
    class DATASET:
        MODE = 'fusion'

        SENTINEL1_BANDS = [0, 1]  # VV and VH bands  
        SENTINEL2_BANDS = [2, 3, 4, 5, 6, 7]  # 6 S2 bands 
 
    
    class MODEL:
        IN_CHANNELS = 8  # 2 S1 bands + 6 S2 bands 
        # IN_CHANNELS = 15
        OUT_CHANNELS = 2  # Binary classification (flood/no-flood)
        TOPOLOGY = [32, 64, 128, 256, 512]
 
        

 
 



 