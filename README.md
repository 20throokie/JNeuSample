# JNeuSample

*NeuSample implemented on Jittor*

*Replace coarse-to-fine, hierarchical sampling procedure by a neural sample field*

## Reference 

Paper: [NeuSample: Neural Sample Field for Efficient View Synthesis](https://arxiv.org/pdf/2111.15552.pdf)  

## Pipeline
![图片](/imgs/pipeline.png)
## Sample Field
<p align = "center">
<img src="./imgs/samplefield.png" width="600px" align="middle" />  
</p>

## Training 

<details>
  <pre><code>
    mpirun -np 4 python main.py --stage train --obj_class lego
  </code></pre>
</details>

Some models are provided in folder **pretrained**
## Comparison
Experiment conducted on one RTX3090 GPU takes around 18 hours for each class.  

  
<img src="./imgs/table1.png" width="1000px">  
<img src="./imgs/table2.png" width="1000px">  
<img src="./imgs/table3.png" width="1000px">  
   
* “Inf. Cost” denotes the relative inference time cost compared with NeRF. <br>
* "Ne" of NeuSample denotes the sample number of the extracted sample field.  

***   




