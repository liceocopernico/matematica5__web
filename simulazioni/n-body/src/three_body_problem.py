import random
import csv
import numpy as np
import itertools
import pyvista as pv


class CBody():
    
    def __init__(self,pos:list,mass:float,radius:float,color:tuple,velocity:list):
        
        self.mass=mass
        self.__pos=np.array(pos,dtype=np.float64)
        self.color=color
        self.velocity=np.array(velocity,dtype=np.float64)
        self.radius=radius
        self.p=self.mass*self.velocity
        self.object=pv.Sphere(radius=self.radius)
        self.orbit:list[np.ndarray]=[]
        
    @property
    def pos(self):
        return self.__pos
    
    @pos.setter
    def pos(self,position):
        self.__pos=position
        self.orbit.append(position)
 
    def set_random_position(self,x:tuple,y:tuple,z:tuple):
        self.pos=np.array([random.uniform(x[0],x[1]),random.uniform(y[0],y[1]),random.uniform(z[0],z[1])],dtype=np.float64)
    def set_random_velocity(self,x:tuple,y:tuple,z:tuple):
        self.velocity=np.array([random.uniform(x[0],x[1]),random.uniform(y[0],y[1]),random.uniform(z[0],z[1])],dtype=np.float64)

   
class BodySystem():
    def __init__(self,data_file:str,dt:float=0.001,G:float=1.0,plotter:pv.Plotter=None):
        file = open(f"data/{data_file}.csv")
        csvreader = csv.reader(file)
        header=next(csvreader)      
        self.bodies:list[CBody]=[]
        self.actors= []
        self.dt=dt
        self.G=G
        self.time=0.0
        self.pl=plotter

        for obj in csvreader:
            mass=float(obj[0])
            position=[float(obj[1]),float(obj[2]),float(obj[3])]
            velocity=[float(obj[4]),float(obj[5]),float(obj[6])]
            radius=float(obj[7])
            new_body=CBody(position,mass,radius,(random.random(),random.random(),random.random()),velocity)
            self.bodies.append(new_body)
            self.actors.append(self.pl.add_mesh(new_body.object,color=new_body.color,render=False))
        
    def get_cm_position(self):
        cm=np.array([0.0,0.0,0.0],dtype=np.float64)
        total_mass=0.0
        for star in self.bodies:
            cm+=star.mass*star.pos
            total_mass+=star.mass
        return cm/total_mass  
    
    def translate_to_cm(self,center_of_mass):
        for i,star in enumerate(self.bodies):
            star.pos=star.pos-center_of_mass   
            self.actors[i].position=star.pos.tolist()
            
    def get_cm_velocity(self):
        vcm=np.array([0.0,0.0,0.0],dtype=np.float64)
        total_mass=0.0
        for star in self.bodies:
            vcm+=star.mass*star.velocity    
            total_mass+=star.mass
        return vcm/total_mass
    
    def do_galileo_transform(self,vcm):
        for star in self.bodies:
            star.velocity=star.velocity-vcm
            star.p=star.mass*star.velocity
    
   
    def init(self):
        self.time=0.0
        self.translate_to_cm(self.get_cm_position())
        self.do_galileo_transform(self.get_cm_velocity())
        print("Initial conditions set. Center of mass at origin and total momentum zero.")
        print(f"CM postion:{self.get_cm_position()}, CM velocity: {self.get_cm_velocity()}")
    
    def pstep(self, star1, star2,delta_t):
        rvector = star1.pos-star2.pos
        r=np.sqrt(rvector.dot(rvector))
        force_vector = -self.G * star1.mass * star2.mass * rvector / r**3
        star1.p = star1.p + force_vector*delta_t
        star2.p = star2.p - force_vector*delta_t

    def init_simulation(self):
        for pair in itertools.combinations(self.bodies,2):
            self.pstep(pair[0],pair[1],0.5*self.dt)
    
    def simulation_step(self,step):
        self.time+=self.dt
                      
        for i,star in enumerate(self.bodies):
                star.pos = star.pos + star.p/star.mass * self.dt
                self.actors[i].position=star.pos
                spline=pv.Spline(np.array(star.orbit),100)
                self.pl.add_mesh(spline,color=star.color,render=False)
        for pair in itertools.combinations(self.bodies,2):
                    self.pstep(pair[0],pair[1],self.dt)       
                

pl = pv.Plotter()

stars = BodySystem("infinito",dt=0.01,G=1.0,plotter=pl)

stars.init()

stars.init_simulation()

pl.add_timer_event(max_steps=3000, duration=2, callback=stars.simulation_step)

cpos = pv.CameraPosition(
    position=(0.0, 0.0, 10.0), focal_point=(0.0, 0.0, 0.0), viewup=(0.0, 1.0, 0.0)
)
pl.show(cpos=cpos)



