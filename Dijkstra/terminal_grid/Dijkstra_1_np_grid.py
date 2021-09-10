
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import time



class astar():
    def __init__(self, obstacle_map, sx, sy, gx, gy):
        

        self.sx = sx
        self.sy = sy
        self.gx = gx
        self.gy = gy
        
        self.obstacle_map = obstacle_map 

        self.costmap = np.zeros_like(self.obstacle_map)
        self.typemap = np.zeros_like(self.obstacle_map)

        self.typemap[ self.obstacle_map == 1 ] = 1
        self.typemap[ self.sy , self.sx ] = 3

        # self.typemap[ self.sy+5 , self.sx+9 ] = 3
        # self.typemap[ self.sy+6 , self.sx+2 ] = 3

        self.reached_goal = False

        self.obstacle_map[sy,sx] = 77
        self.obstacle_map[gy,gx] = 33

        print(self.obstacle_map)

        self.clear_to_move, self.cannot_moveto = [], [] 
        

        # self.costmap[sy,sx] = 0  # cost at starting point 

        self.get_motion_model()
        
        self.obstacle_map_view = self.obstacle_map.copy()

        # self.obstacle_map_view = self.binary2color(self.obstacle_map.copy())
        # self.cvshow_larger(self.obstacle_map_view, 30, 0)

        self.planning()


    def planning(self):
        open_nodes = dict()
        close_nodes = dict()

        open_nodes[(self.sx, self.sy)] = [ self.sx, self.sy, 0]

        self.reached_goal = False

        while self.reached_goal == False:

            type3xy = np.where(self.typemap == 3)
            centers_in_this_round = np.array((type3xy[0].tolist(),type3xy[1].tolist())).T
            # print('centers_in_this_round  \n',centers_in_this_round)

            if centers_in_this_round.shape[0] < 1:
                print('There is no grid to check !!!')
            
            # for each grid appeared as centers 
            num_of_centers = centers_in_this_round.shape[0]
            for i in range(num_of_centers):
                # loop ove the 8 grid arounf it
                x = centers_in_this_round[i][1]
                y = centers_in_this_round[i][0]

                self.update_typemap_per_center(x, y)

            # print('\ntypemap')
            # print(self.typemap)

            type4xy = np.where(self.typemap == 4)
            points_to_check = np.array((type4xy[0].tolist(),type4xy[1].tolist())).T
            # print('\npoints_to_check  \n',points_to_check)

            if points_to_check.shape[0] < 1:
                print('There is no grid to check !!!')
            
            # for each grid need to compute cost 
            num_of_points = points_to_check.shape[0]
            for i in range(num_of_points):
                #  sda
                x = points_to_check[i][1]
                y = points_to_check[i][0] 
                
                self.update_cost_this_grid(x, y)

            # print('\ncostmap')
            # print(self.costmap)

            self.typemap[self.typemap == 3] = 2
            self.typemap[self.typemap == 4] = 3

            # print('\n\n\n\n\n')
            # time.sleep(1)

        print('\ncostmap')
        print(self.costmap)
        
        print(self.costmap[self.gy, self.gx])

        self.costmap[self.costmap == 0] = 9999

        # find the path from costmap
        backed = False
        xys = set()
        # xys = []
        [x,y] = self.find_smallest_cost(self.gx, self.gy)
        while backed == False:
            [x,y] = self.find_smallest_cost(x, y)
            xys.add((x,y))
            # xys.append([x,y])
            print(x,y)
            if abs(x-self.sx) + abs(y-self.sy) <= 1.5:
                backed = True

            # time.sleep(0.1)
        # xys.add((self.sx, self.sy))
        xys = list(xys)
        # print(len(xys))
        # print(xys)
        for i in range(len(xys)):
            # print(xys[i])
            x = xys[i][0]
            y = xys[i][1]
            self.obstacle_map_view[y,x] = 7

        print(self.obstacle_map_view)



    def update_typemap_per_center(self, x,y):
        for i in self.motion:
            newx = x + i[0]
            newy = y + i[1]

            if self.obstacle_map[newy, newx] == 1:
                self.typemap[newy, newx] = 1
            elif self.typemap[newy, newx] == 0:
                self.typemap[newy, newx] = 4


    def update_cost_this_grid(self, x, y):
        parent_costs = []
        cost = []

        if self.gy == y and self.gx == x:
            self.reached_goal = True

        for i in self.motion:
            newx = x + i[0]
            newy = y + i[1]

            

            if self.typemap[newy, newx] == 3:
                cost.append( self.costmap[newy,newx] + i[2] )
                parent_costs.append( self.costmap[newy,newx] )
            # elif self.typemap[newy, newx] != 2:
            #     self.typemap[newy, newx] = 4
        # print('parent costs', parent_costs)
        smallest_parent_cost = min(parent_costs)
        smallest_cost = min(cost)
        
        self.costmap[y,x] = smallest_cost
        # return smallest_parent_cost

    def find_smallest_cost(self, x, y):
        costs = []
        valuable_grids = []
        for i in self.motion:
            newx = x + i[0]
            newy = y + i[1]

            # if self.costmap[newy, newx] != 0:
            costs.append(self.costmap[newy, newx])
            valuable_grids.append( [newx, newy] )
        min_c = min(costs)
        min_c_i = costs.index(min_c)
        return valuable_grids[min_c_i]
        



    def update_one_node(self, x, y, cost):
        clear_to_move_in_update = []
        cannot_moveto_in_update = []
        for i in self.motion:
            newx = x + i[0]
            newy = y + i[1]
            newcost = cost + i[2]

            # print(self.obstacle_map[newy, newx])

            if self.obstacle_map[newy, newx] == 0:
                newcost = newcost
                clear_to_move_in_update.append([newx, newy, newcost])
            elif self.obstacle_map[newy,newx] == 1:
                newcost = -1 
                cannot_moveto_in_update.append([newx, newy, newcost])


        cannot_moveto_in_update.append([x, y, cost])

        return clear_to_move_in_update, cannot_moveto_in_update

                


    def get_motion_model(self):
        # dx, dy, cost
        self.motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

       
    def binary2color(self, img):
        img[img==0] = 255
        img[img==1] = 0
        img[img==2] = 55
        img[img==3] = 150
        
        return img.astype(np.uint8)

    def cvshow_larger(self, img, ratio, t):
        original_y = img.shape[0]
        original_x = img.shape[1]
        enlarged = cv2.resize(img, (original_x*ratio, original_y*ratio),interpolation = cv2.INTER_NEAREST)
        print('img size', img.shape)
        print('large size', enlarged.shape)
        cv2.imshow('plan', enlarged)
        cv2.waitKey(0)








def main():
    # start and goal position
    
    sx = 2
    sy = 10
    
    gx = 13
    gy = 3


    obstacle_map = np.array(
        [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        ]
    )

    

    # plt.plot(sx, sy, "og")
    
    # plt.plot(gx, gy, "xb")

    # plt.grid(True)
    # plt.axis("equal")

    astarplanner = astar(obstacle_map, sx, sy, gx, gy)

    # plt.show()


if __name__ == '__main__':
    main()
    


