


resnet74=12*2*[[16,16,32,3,1]]+[[16,32,16,3,2],[16,32,16,1,2]]+11*2*[[32,32,16,3,1]]+[[32,64,8,3,2],[32,64,8,1,2]]+11*2*[[64,64,8,3,1]]
resnet110=18*2*[[16,16,32,3,1]]+[[16,32,16,3,2],[16,32,16,1,2]]+17*2*[[32,32,16,3,1]]+[[32,64,8,3,2],[32,64,8,1,2]]+17*2*[[64,64,8,3,1]]
resnet164=27*2*[[16,16,32,3,1]]+[[16,32,16,3,2],[16,32,16,1,2]]+26*2*[[32,32,16,3,1]]+[[32,64,8,3,2],[32,64,8,1,2]]+26*2*[[64,64,8,3,1]]
mbv2=[[32,32*6,32,1,1],[1,32*6,32,3,1],[32*6,16,32,1,1],\

      [16,16*6,32,1,1],[1,16*6,32,3,1],[16*6,24,32,1,1],\
      [24,24*6,32,1,1],[1,24*6,32,3,1],[24*6,24,32,1,1],\
      
      [24,24*6,32,1,1],[1,24*6,16,3,1],[24*6,32,16,1,1],\
      [32,32*6,16,1,1],[1,32*6,16,3,1],[32*6,32,16,1,1],\
      [32,32*6,16,1,1],[1,32*6,16,3,1],[32*6,32,16,1,1],\
      
      [32,32*6,16,1,1],[1,32*6, 8,3,1],[32*6,64, 8,1,1],\
      [64,64*6, 8,1,1],[1,64*6, 8,3,1],[64*6,64, 8,1,1],\
      [64,64*6, 8,1,1],[1,64*6, 8,3,1],[64*6,64, 8,1,1],\
      [64,64*6, 8,1,1],[1,64*6, 8,3,1],[64*6,64, 8,1,1],\
      
      [64,64*6, 8,1,1],[1,64*6, 8,3,1],[64*6,96, 8,1,1],\
      [96,96*6, 8,1,1],[1,96*6, 8,3,1],[96*6,96, 8,1,1],\
      [96,96*6, 8,1,1],[1,96*6, 8,3,1],[96*6,96, 8,1,1],\
      
      [96,96*6, 8,1,1],[1,96*6, 4,3,1],[96*6,160,4,1,1],\
      [160,160*6, 4,1,1],[1,160*6, 4,3,1],[160*6,160,4,1,1],\
      [160,160*6, 4,1,1],[1,160*6, 4,3,1],[160*6,160,4,1,1],\
      
      [160,160*6, 4,1,1],[1,160*6, 4,3,1],[160*6,320,4,1,1],\
      
    ]
mbv2_dw=[0,1,0,\
         
         0,1,0,\
         0,1,0,\
         
         0,1,0,\
         0,1,0,\
         0,1,0,\
         
         0,1,0,\
         0,1,0,\
         0,1,0,\
         0,1,0,\
         
         0,1,0,\
         0,1,0,\
         0,1,0,\
         
         0,1,0,\
         0,1,0,\
         0,1,0,\
         
         0,1,0,\
        ]