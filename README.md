# Python Package
numpy == 1.19.5  
opencv-python == 3.2.0.8  
Pillow == 6.2.0  
pandas == 1.1.5  

# To run this file (This may need 1 min for one loop)
unzip yaleface_raw_images.rar to the path 
## Method 1 
> cd <path_to_the_P1>  
> python main.py  
if you have python2 installed you may need to run:  
> python3 main.py  
## Method 2 to adjust parameters
> cd <path_to_the_P1>  
> python main.py --loop 1 --k_knn 3 --lambd 0 --hidden_neurons 24 --test_size 4  

### Writer: Jason S
