import numpy as np
import math
from PIL import Image, ImageDraw

class random_mask_generator():
    """
    Generate random mask by defiining generation parameters.

    Output: Mask of size image input size (W,H)
    """

    def __init__(self,image_size = (256,256),vertex = 5, brush_width = (5,25), mean_angle = 2*math.pi/5, angle_range = 2*math.pi / 15, mask_floor = 10, mask_ceiling = 20, max_iter=10):
        self.w= image_size[0]
        self.h= image_size[1]
        #self.min_vertex = vertex[0]
        self.max_vertex = vertex
        self.min_brush_width = brush_width[0]
        self.max_brush_width = brush_width[1]
        self.mean_angle = mean_angle
        self.angle_range = angle_range
        self.mask_percent_floor = mask_floor
        self.mask_percent_ceiling = mask_ceiling
        self.max_iter = max_iter


    def image_generator(self):
        """
        Initiate Mask Image using input image dimension
        """
        image = Image.new('L',(self.w,self.h))
        return image
    
    def mask_percent(self, mask):
        """
        Calculate mask percentage
        Input: Mask
        Output: Mask Percentage
        """
        w,h = mask.shape
        mask_percent = round(np.sum(mask>0)*100/(w*h),2)
        return mask_percent
    
    def mask_generator(self):
        """
        Generate random mask.

        output: Random mask generated as per defined parameters
        """
        
        w,h = self.w, self.h
        average_radius = math.sqrt(h*h+w*w) / 8
        #num_vertex = np.random.randint(self.min_vertex,self.max_vertex)
        num_vertex = self.max_vertex
        #print(f'No. of Vertex: {num_vertex}')
        brush_width = np.random.uniform(self.min_brush_width, self.max_brush_width)
        #print(f'Brush Width: {brush_width}')


        # Mask Initiation
        mask = self.image_generator()
        draw = ImageDraw.Draw(mask)
        angles = []
        vertices = []
        mask_prev = np.asarray(mask, np.float32)
        mask_percent_prev = 0

        for j in range(np.random.randint(1, self.max_iter)):
            angles = []
            vertices = []
            start_x = np.random.randint(0,w)
            start_y = np.random.randint(0,h)
            vertices.append((int(start_x), int(start_y)))
            for i in range(num_vertex):

                if i%2==0:
                    angle_min = self.mean_angle - np.random.uniform(0, self.angle_range)
                    angle_max = self.mean_angle + np.random.uniform(0, self.angle_range)

                    angles.append(2*math.pi - np.random.uniform(angle_min,angle_max))
                    
                else:
                    angles.append(np.random.uniform(angle_min,angle_max))
                r = np.clip(np.random.normal(loc=average_radius, scale=average_radius//2),0, 2*average_radius)
                new_x = int(vertices[-1][0] + r*math.cos(angles[i]))
                new_y = int(vertices[-1][1] + r*math.sin(angles[i]))
                
                #Drawing lines and ellpise shaped masks between previous and current vertices
                draw.line([vertices[-1], (new_x,new_y)], fill='white', width=self.max_brush_width)
                draw.ellipse([(vertices[-1][0]- self.max_brush_width//2,vertices[-1][1]+ self.max_brush_width//2),
                            (new_x- self.max_brush_width//2,new_y+ self.max_brush_width//2)],fill='white', outline='white')
                vertices.append(((new_x), (new_y)))
                
                #Checking mask percentage
                mask_iter = np.asarray(mask, np.float32)
                mask_iter_percent = self.mask_percent(mask_iter)
                
#                 if mask_iter_percent<self.mask_percent_floor:  
#                     mask_prev = mask_iter
#                     mask_percent_prev = mask_iter_percent
                if mask_iter_percent<self.mask_percent_floor:  
                    continue
                if mask_iter_percent<self.mask_percent_ceiling:  
                    mask_prev = mask_iter
                    mask_percent_prev = mask_iter_percent
                else:
                    break
            

        #mask.show()
        mask_array = np.asarray(mask_prev, np.float32)
        mask_percent = self.mask_percent(mask_array)
        #print(f'final mask percent is {round(mask_percent,2)}') 
        return mask, mask_array, mask_percent



