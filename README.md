# code TAwTV
***************************************************************************
Cone Beam Computed Tomography Based on Truncated Adaptive Weight Total Variation
***************************************************************************     
 Copyright:   Yuhang Liu, Yi Liu, Pengcheng Zhang, Rongbiao Yan, Lei Wang 
              Wengting Liu, Zhiguo Gui.                 
***************************************************************************
  TAwTV is an improved model based on the classic TV. In this project, 
  TAwTV is used in CT reconstruction. The method of calling the function 
  can be viewed in IncART.cpp.
  
  In addition to TAwTV, POCS_TV.cu also includes TV, TTV, AwTV, and HOTV.
***************************************************************************
  pocs_tv(float* img,float* dst,float alpha,const long* image_size, int maxIter, bool HOTV, bool RwTV, float delta);

  if HOTV is true, pocs_tv is HOTV minimization.
  
  if HOTV is false and RwTV is true, pocs_tv is 3-D standard TV minimization. In this case, if the annotation symbol about TTV is removed, it is TTV minimization.
  
  if HOTV is false and RwTV is false , pocs_tv is TAwTV minimization. In this case, if the annotation symbol about AwTV is removed and the annotation symbol about TAwTV is added, it is AwTV minimization.
  
  "if else" code is very simple, just look at the source code and it will be clear.
***************************************************************************
  More detail can be found in [1]
  
  [1] Yuhang Liu, Yi Liu, Pengcheng Zhang, Rongbiao Yan, Lei Wang Wengting Liu, 
  Zhiguo Gui. Cone-beam Computed Tomography Based on Truncated Adaptive-weight 
  Total Variation[J]. NDT & E International, 2022, 133:102755.
***************************************************************************
 Please cite our paper if you use any part of our source code.
