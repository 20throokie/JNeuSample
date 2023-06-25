# JNeuSample

*NeuSample implemented on Jittor*

*Replace coarse-to-fine, hierarchical sampling procedure by a neural sample field*

## Reference 

Paper: [NeuSample: Neural Sample Field for Efficient View Synthesis](https://arxiv.org/pdf/2111.15552.pdf)  

## Pipeline
![图片](/imgs/pipeline.png)
## Sample Field
<p align = "center">
<img src="./imgs/samplefield.png" width="550px" align="middle" />  
</p>

## Training 

```
mpirun -np 4 python main.py --stage train --obj_class lego
```

Some models are provided in folder **pretrained**
## Comparison
Experiment conducted on one RTX3090 GPU takes around 18 hours for each class.  

<table border="1" width="1200px" cellspacing="20">
<tr>
  <th rowspan="2" align="center" valign="center">Class</th>
  <th colspan="2" align="center" valign="center">PSNR</th>
  <th colspan="2" align="center" valign="center">SSIM</th>
  <th colspan="2" align="center" valign="center">LPIPS</th>
</tr>
<tr>
  <td align="center" valign="center">Paper</td>
  <td align="center" valign="center">This Code</td>
  <td align="center" valign="center">Paper</td>
  <td align="center" valign="center">This Code</td>
  <td align="center" valign="center">Paper</td>
  <td align="center" valign="center">This Code</td>
</tr>
<tr>
  <td align="center" valign="center">Chair</td>
  <td align="center" valign="center">33.02</td>
  <td align="center" valign="center><b>32.46</b></td>
  <td align="center" valign="center">**0.968**</td>
  <td align="center" valign="center">0.961</td>
  <td align="center" valign="center">**0.045**</td>
  <td align="center" valign="center">0.051</td>
</tr>
<tr>
  <td align="center" valign="center">Drums</td>
  <td align="center" valign="center">24.99</td>
  <td align="center" valign="center">25.94</td>
  <td align="center" valign="center">0.924</td>
  <td align="center" valign="center">0.928</td>
  <td align="center" valign="center">0.091</td>
  <td align="center" valign="center">0.093</td>
</tr>

</table>

   


  
***   




