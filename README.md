# JNeuSample

NeuSample implemented on Jittor

**Replace coarse-to-fine, hierarchical sampling procedure by a neural sample field**

## Reference
Paper: [NeuSample: Neural Sample Field for Efficient View Synthesis](https://arxiv.org/pdf/2111.15552.pdf)
## Pipeline
![图片](/imgs/pipeline.png)
## Sample Field
<img src="https://github.com/20throokie/JNeuSample/blob/master/imgs/samplefield.png" width="600px">  

## Training 
> mpirun -np 4 python main.py --stage train

## Comparison
<img src="https://github.com/20throokie/JNeuSample/blob/master/imgs/table1.png" width="1000px">  
<img src="https://github.com/20throokie/JNeuSample/blob/master/imgs/table2.png" width="1000px">  
<img src="https://github.com/20throokie/JNeuSample/blob/master/imgs/table3.png" width="1000px">  
* “Inf. Cost” denotes the relative inference cost compared with NeRF, i.e. time for rendering one image which is measured on one RTX3090 GPU.  <br>
* "Ne" of NeuSample denotes the sample number of the extracted sample field.  



