import heapq
from itertools import permutations

path_value = 0
start = 'START'
def pathfinder(st, nd): 
    
    class Node:
        def __init__(self, name , x, y, z,yaw):
            self.name = name
            self.x = x
            self.y = y
            self.z = z
            self.yaw = yaw
            self.parent = None
            self.g = float('inf')  # Cost to reach this node from the start node
            self.h = 0  # Heuristic cost estimate of the cheapest path from here to the goal
            self.f = float('inf')  # Estimated total cost (g + h)
           

        def __lt__(self, other):
            return self.f < other.f

        def __eq__(self, other):
            return self.x == other.x and self.y == other.y and self.z == other.z and self.yaw == other.yaw

        def __hash__(self):
            return hash((self.x, self.y, self.z))

    def heuristic(node, goal):
        # Using Euclidean distance as heuristic
        return ((node.x - goal.x) ** 2 + (node.y - goal.y) ** 2 + (node.z - goal.z) ** 2) ** 0.5
 
    def a_star_3d(start, goal, max_x, max_y, max_z):
        global path_value
        
        open_set = []
        heapq.heappush(open_set, start)
        start.g = 0
        start.h = heuristic(start, goal)
        start.f = start.h

        closed_set = set()

        while open_set:
            current = heapq.heappop(open_set) 
            path_value+=current.f
            if current == goal:
                
                path = []
                while current.parent:
                    path.append(current)
                    current = current.parent
                return path[::-1]

            closed_set.add(current)

            for neighbor in graph[current]:
                if neighbor in closed_set:
                    continue

                tentative_g_score = current.g + 1  # Assuming a uniform cost

                if tentative_g_score < neighbor.g:
                    neighbor.parent = current
                    neighbor.g = tentative_g_score
                    neighbor.h = heuristic(neighbor, goal)
                    neighbor.f = neighbor.g + neighbor.h

                   # print(open_set," : ",current.name," :: ",neighbor.name," : ",neighbor.f)
                    if neighbor not in open_set: 
                        heapq.heappush(open_set, neighbor)
                        

        return None

    nodes = {
        'START': Node('START', 1.5, 1.5, 2, 0),
        '1B': Node('1B', 7.01, 6, 1.4, 1),
        '1F': Node('1F', 1, 6, 1.4, 0),
        '2B': Node('2B', 7.01, 6, 4.2, 1),
        '2F': Node('2F', 1, 6, 4.2, 0),
        '3B': Node('3B', 7.01, 6, 6.9, 1),
        '3F': Node('3F', 1, 6, 6.9, 0),
        '4B': Node('4B', 7.01, 13.5, 1.4, 1),
        '4F': Node('4F', 1, 13.5, 1.4, 0),
        '5B': Node('5B', 7.01, 13.5, 4.2, 1),
        '5F': Node('5F', 1, 13.5, 4.2, 0),
        '6B': Node('6B', 7.01, 13.5, 6.9, 1),
        '6F': Node('6F', 1, 13.5, 6.9, 0),
        '7B': Node('7B', 7.01, 21, 1.4, 1),
        '7F': Node('7F', 1, 21, 1.4, 0),
        '8B': Node('8B', 7.01, 21, 4.2, 1),
        '8F': Node('8F', 1, 21, 4.2, 0),
        '9B': Node('9B', 7.01, 21, 6.9, 1),
        '9F': Node('9F', 1, 21, 6.9, 0),
        '10B': Node('10B', 13.01, 6, 1.4, 1),
        '10F': Node('10F', 6.95, 6, 1.4, 0),
        '11B': Node('11B', 13.01, 6, 4.2, 1),
        '11F': Node('11F', 6.95, 6, 4.2, 0),
        '12B': Node('12B', 13.01, 6, 6.9, 1),
        '12F': Node('12F', 6.95, 6, 6.9, 0),
        '13B': Node('13B', 13.01, 13.5, 1.4, 1),
        '13F': Node('13F', 6.95, 13.5, 1.4, 0),
        '14B': Node('14B', 13.01, 13.5, 4.2, 1),
        '14F': Node('14F', 6.95, 13.5, 4.2, 0),
        '15B': Node('15B', 13.01, 13.5, 6.9, 1),
        '15F': Node('15F', 6.95, 13.5, 6.9, 0),
        '16B': Node('16B', 13.01, 21, 1.4, 1),
        '16F': Node('16F', 6.95, 21, 1.4, 0),
        '17B': Node('17B', 13.01, 21, 4.2, 1),
        '17F': Node('17F', 6.95, 21, 4.2, 0),
        '18B': Node('18B', 13.01, 21, 6.9, 1),
        '18F': Node('18F', 6.95, 21, 6.9, 0),
        '19B': Node('19B', 19, 6, 1.4, 1),
        '19F': Node('19F', 12.99, 6, 1.4, 0),
        '20B': Node('20B', 19, 6, 4.2, 1),
        '20F': Node('20F', 12.99, 6, 4.2, 0),
        '21B': Node('21B', 19, 6, 6.9, 1),
        '21F': Node('21F', 12.99, 6, 6.9, 0),
        '22B': Node('22B', 19, 13.5, 1.4, 1),
        '22F': Node('22F', 12.99, 13.5, 1.4, 0),
        '23B': Node('23B', 19, 13.5, 4.2, 1),
        '23F': Node('23F', 12.99, 13.5, 4.2, 0),
        '24B': Node('24B', 19, 13.5, 6.9, 1),
        '24F': Node('24F', 12.99, 13.5, 6.9, 0),
        '25B': Node('25B', 19, 21, 1.4, 1),
        '25F': Node('25F', 12.99, 21, 1.4, 0),
        '26B': Node('26B', 19, 21, 4.2, 1),
        '26F': Node('26F', 12.99, 21, 4.2, 0),
        '27B': Node('27B', 19, 21, 6.9, 1),
        '27F': Node('27F', 12.99, 21, 6.9, 0),

        'AL71': Node('AL71', 1.5, 25.5, 1.6, 0),
        'AL72': Node('AL72', 6.9, 25.5, 1.6, 0),
        'AL81': Node('AL81', 1.5, 25.5, 4.2, 0),
        'AL82': Node('AL82', 6.9, 25.5, 4.2, 0),
        'AL91': Node('AL91', 1.5, 25.5, 6.9, 0),
        'AL92': Node('AL92', 6.9, 25.5 ,6.9, 0),
        'AR11': Node('AR11', 1, 1.5, 1.6, 0),
        'AR12': Node('AR12', 7, 1.5, 1.6, 0),
        'AR21': Node('AR21', 1, 1.5, 4.2, 0),
        'AR22': Node('AR22', 7, 1.5, 4.2, 0),
        'AR31': Node('AR31', 1, 1.5, 6.9, 0),
        'AR32': Node('AR32', 6.9, 1.5, 6.9, 0),
        'BL71': Node('BL71', 6.91, 25.5, 1.6, 0),
        'BL72': Node('BL72', 12.9, 25.5, 1.6, 1),
        'BL81': Node('BL81', 6.91, 25.5, 4.2, 0),
        'BL82': Node('BL82', 12.9, 25.5, 4.2, 1),
        'BL91': Node('BL91', 6.91, 25.5, 6.9, 0),
        'BL92': Node('BL92', 12.9, 25.5, 6.9, 1),
        'BR11': Node('BR11', 6.91, 1.5, 1.6, 0),
        'BR12': Node('BR12', 12.9, 1.5, 1.6, 1),
        'BR21': Node('BR21', 6.91, 1.5, 4.2, 0),
        'BR22': Node('BR22', 12.9, 1.5, 4.2, 1),
        'BR31': Node('BR31', 6.91, 1.5, 6.9, 0),
        'BR32': Node('BR32', 12.9, 1.5, 6.9, 1),
        'CL71': Node('CL71', 13, 25.5, 1.6, 1),
        'CL72': Node('CL72', 19, 25.5, 1.6, 1),
        'CL81': Node('CL81', 13, 25.5, 4.2, 1),
        'CL82': Node('CL82', 19, 25.5, 4.2, 1),
        'CL91': Node('CL91', 13, 25.5, 6.9, 1),
        'CL92': Node('CL92', 19, 25.5, 6.9, 1),
        'CR11': Node('CR11', 13, 1.5, 1.6, 1),
        'CR12': Node('CR12', 19, 1.5, 1.6, 1),
        'CR21': Node('CR21', 13, 1.5, 4.2, 1),
        'CR22': Node('CR22', 19, 1.5, 4.2, 1),
        'CR31': Node('CR31', 13, 1.5, 6.9, 1),
        'CR32': Node('CR32', 19, 1.5, 6.9, 1),
        'AR13': Node('AR13', 4, 1.5, 1.6, 0),
        'AR23': Node('AR13', 4, 1.5, 4.2, 0),
        'AR33': Node('AR13', 4, 1.5, 6.9, 0),
        'BR13': Node('BR13', 10, 1.5, 1.6, 0),
        'BR23': Node('BR13', 10, 1.5, 4.2, 0),
        'BR33': Node('BR13', 10, 1.5, 6.9, 0),
        'CR13': Node('CR13', 16, 1.5, 1.6, 1),
        'CR23': Node('CR13', 16, 1.5, 4.2, 1),
        'CR33': Node('CR13', 16, 1.5, 6.9, 1),
        'AL73': Node('AL73', 4, 25.5, 1.6, 0),
        'AL83': Node('AL83', 4, 25.5, 4.2, 0),
        'AL93': Node('AL93', 4, 25.5, 6.9, 0),
        'BL73': Node('BL73', 10, 25.5, 1.6,0),
        'BL83': Node('BL83', 10, 25.5, 4.2, 0),
        'BL93': Node('BL93', 10, 25.5, 6.9, 0),
        'CL73': Node('CL73', 16, 25.5, 1.6, 1),
        'CL83': Node('CL83', 16, 25.5, 4.2, 1),
        'CL93': Node('CL93', 16, 25.5, 6.9, 1)
    }

    graph = {
        nodes['START']: [nodes['1F'], nodes['2F'], nodes['3F'], nodes['4F'], nodes['5F'], nodes['6F'], nodes['7F'],nodes['8F'], nodes['9F'], nodes['AR11'], nodes['AR21'], nodes['AR31'], nodes['AL71'], nodes['AL81'], nodes['AL91'],nodes['BR13'], nodes['BR23'], nodes['BR33']],
        nodes['1F']: [nodes['2F'], nodes['3F'], nodes['4F'], nodes['5F'], nodes['6F'], nodes['7F'], nodes['8F'], nodes['9F'], nodes['AR11'], nodes['AR21'], nodes['AR31'], nodes['AL71'], nodes['AL81'], nodes['AL91'],nodes['START']],
        nodes['2F']: [nodes['1F'], nodes['3F'], nodes['4F'], nodes['5F'], nodes['6F'], nodes['7F'], nodes['8F'], nodes['9F'], nodes['AR11'], nodes['AR21'], nodes['AR31'], nodes['AL71'], nodes['AL81'], nodes['AL91'],nodes['START']],
        nodes['3F']: [nodes['1F'], nodes['2F'], nodes['4F'], nodes['5F'], nodes['6F'], nodes['7F'], nodes['8F'], nodes['9F'], nodes['AR11'], nodes['AR21'], nodes['AR31'], nodes['AL71'], nodes['AL81'], nodes['AL91'],nodes['START']],
        nodes['4F']: [nodes['1F'], nodes['2F'], nodes['3F'], nodes['5F'], nodes['6F'], nodes['7F'], nodes['8F'], nodes['9F'], nodes['AR11'], nodes['AR21'], nodes['AR31'], nodes['AL71'], nodes['AL81'], nodes['AL91'],nodes['START']],
        nodes['5F']: [nodes['1F'], nodes['2F'], nodes['3F'], nodes['4F'], nodes['6F'], nodes['7F'], nodes['8F'], nodes['9F'], nodes['AR11'], nodes['AR21'], nodes['AR31'], nodes['AL71'], nodes['AL81'], nodes['AL91'],nodes['START']],
        nodes['6F']: [nodes['1F'], nodes['2F'], nodes['3F'], nodes['4F'], nodes['5F'], nodes['7F'], nodes['8F'], nodes['9F'], nodes['AR11'], nodes['AR21'], nodes['AR31'], nodes['AL71'], nodes['AL81'], nodes['AL91'],nodes['START']],
        nodes['7F']: [nodes['1F'], nodes['2F'], nodes['3F'], nodes['4F'], nodes['5F'], nodes['6F'], nodes['8F'], nodes['9F'], nodes['AL71'], nodes['AL81'], nodes['AL91'], nodes['AR11'], nodes['AR21'], nodes['AR31'],nodes['START']],
        nodes['8F']: [nodes['1F'], nodes['2F'], nodes['3F'], nodes['4F'], nodes['5F'], nodes['6F'], nodes['7F'], nodes['9F'], nodes['AL71'], nodes['AL81'], nodes['AL91'], nodes['AR11'], nodes['AR21'], nodes['AR31'],nodes['START']],
        nodes['9F']: [nodes['1F'], nodes['2F'], nodes['3F'], nodes['4F'], nodes['5F'], nodes['6F'], nodes['7F'], nodes['8F'], nodes['AL71'], nodes['AL81'], nodes['AL91'], nodes['AR11'], nodes['AR21'], nodes['AR31'],nodes['START']],
#
        nodes['10F']: [nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F'], nodes['BR11'], nodes['BR21'], nodes['BR31'], nodes['AR12'], nodes['AR22'], nodes['AR32'], nodes['BL71'], nodes['BL81'], nodes['BL91'], nodes['AL72'], nodes['AL82'], nodes['AL92'],nodes['1B'], nodes['2B'], nodes['3B'], nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B']],
        nodes['11F']: [nodes['10F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F'], nodes['BR11'], nodes['BR21'], nodes['BR31'], nodes['AR12'], nodes['AR22'], nodes['AR32'], nodes['BL71'], nodes['BL81'], nodes['BL91'], nodes['AL72'], nodes['AL82'], nodes['AL92'],nodes['1B'], nodes['2B'], nodes['3B'], nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B']],
        nodes['12F']: [nodes['10F'], nodes['11F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F'], nodes['BR11'], nodes['BR21'], nodes['BR31'], nodes['AR12'], nodes['AR22'], nodes['AR32'], nodes['BL71'], nodes['BL81'], nodes['BL91'], nodes['AL72'], nodes['AL82'], nodes['AL92'],nodes['1B'], nodes['2B'], nodes['3B'], nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B']],
        nodes['13F']: [nodes['10F'], nodes['11F'], nodes['12F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F'], nodes['BR11'], nodes['BR21'], nodes['BR31'], nodes['AR12'], nodes['AR22'], nodes['AR32'], nodes['BL71'], nodes['BL81'], nodes['BL91'], nodes['AL72'], nodes['AL82'], nodes['AL92'],nodes['1B'], nodes['2B'], nodes['3B'], nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B']],
        nodes['14F']: [nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F'], nodes['BR11'], nodes['BR21'], nodes['BR31'], nodes['AR12'], nodes['AR22'], nodes['AR32'], nodes['BL71'], nodes['BL81'], nodes['BL91'], nodes['AL72'], nodes['AL82'], nodes['AL92'],nodes['1B'], nodes['2B'], nodes['3B'], nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B']],
        nodes['15F']: [nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['16F'], nodes['17F'], nodes['18F'], nodes['BR11'], nodes['BR21'], nodes['BR31'], nodes['AR12'], nodes['AR22'], nodes['AR32'], nodes['BL71'], nodes['BL81'], nodes['BL91'], nodes['AL72'], nodes['AL82'], nodes['AL92'],nodes['1B'], nodes['2B'], nodes['3B'], nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B']],
        nodes['16F']: [nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['17F'], nodes['18F'], nodes['BR11'], nodes['BR21'], nodes['BR31'], nodes['AR12'], nodes['AR22'], nodes['AR32'], nodes['BL71'], nodes['BL81'], nodes['BL91'], nodes['AL72'], nodes['AL82'], nodes['AL92'],nodes['1B'], nodes['2B'], nodes['3B'], nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B']],
        nodes['17F']: [nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['18F'], nodes['BR11'], nodes['BR21'], nodes['BR31'], nodes['AR12'], nodes['AR22'], nodes['AR32'], nodes['BL71'], nodes['BL81'], nodes['BL91'], nodes['AL72'], nodes['AL82'], nodes['AL92'],nodes['1B'], nodes['2B'], nodes['3B'], nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B']],
        nodes['18F']: [nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['BR11'], nodes['BR21'], nodes['BR31'], nodes['AR12'], nodes['AR22'], nodes['AR32'], nodes['BL71'], nodes['BL81'], nodes['BL91'], nodes['AL72'], nodes['AL82'], nodes['AL92'],nodes['1B'], nodes['2B'], nodes['3B'], nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B']],

        nodes['1B']: [nodes['2B'], nodes['3B'], nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B'], nodes['BR11'], nodes['BR21'], nodes['BR31'], nodes['AR12'], nodes['AR22'], nodes['AR32'], nodes['BL71'], nodes['BL81'], nodes['BL91'], nodes['AL72'], nodes['AL82'], nodes['AL92'], nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F']],
        nodes['2B']: [nodes['1B'], nodes['3B'], nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B'], nodes['BR11'], nodes['BR21'], nodes['BR31'], nodes['AR12'], nodes['AR22'], nodes['AR32'], nodes['BL71'], nodes['BL81'], nodes['BL91'], nodes['AL72'], nodes['AL82'], nodes['AL92'], nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F']],
        nodes['3B']: [nodes['1B'], nodes['2B'], nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B'], nodes['BR11'], nodes['BR21'], nodes['BR31'], nodes['AR12'], nodes['AR22'], nodes['AR32'], nodes['BL71'], nodes['BL81'], nodes['BL91'], nodes['AL72'], nodes['AL82'], nodes['AL92'], nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F']],
        nodes['4B']: [nodes['1B'], nodes['2B'], nodes['3B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B'], nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F'], nodes['BR11'], nodes['BR21'], nodes['BR31'], nodes['AR12'], nodes['AR22'], nodes['AR32'], nodes['BL71'], nodes['BL81'], nodes['BL91'], nodes['AL72'], nodes['AL82'], nodes['AL92']],
        nodes['5B']: [nodes['1B'], nodes['2B'], nodes['3B'], nodes['4B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B'], nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F'], nodes['BR11'], nodes['BR21'], nodes['BR31'], nodes['AR12'], nodes['AR22'], nodes['AR32'], nodes['BL71'], nodes['BL81'], nodes['BL91'], nodes['AL72'], nodes['AL82'], nodes['AL92']],
        nodes['6B']: [nodes['1B'], nodes['2B'], nodes['3B'], nodes['4B'], nodes['5B'], nodes['7B'], nodes['8B'], nodes['9B'], nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F'], nodes['BR11'], nodes['BR21'], nodes['BR31'], nodes['AR12'], nodes['AR22'], nodes['AR32'], nodes['BL71'], nodes['BL81'], nodes['BL91'], nodes['AL72'], nodes['AL82'], nodes['AL92']],
        nodes['7B']: [nodes['1B'], nodes['2B'], nodes['3B'], nodes['4B'], nodes['5B'], nodes['6B'], nodes['8B'], nodes['9B'], nodes['BR11'], nodes['BR21'], nodes['BR31'], nodes['AR12'], nodes['AR22'], nodes['AR32'], nodes['BL71'], nodes['BL81'], nodes['BL91'], nodes['AL72'], nodes['AL82'], nodes['AL92'], nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F']],
        nodes['8B']: [nodes['1B'], nodes['2B'], nodes['3B'], nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['9B'], nodes['BR11'], nodes['BR21'], nodes['BR31'], nodes['AR12'], nodes['AR22'], nodes['AR32'], nodes['BL71'], nodes['BL81'], nodes['BL91'], nodes['AL72'], nodes['AL82'], nodes['AL92'], nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F']],
        nodes['9B']: [nodes['1B'], nodes['2B'], nodes['3B'], nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['BR11'], nodes['BR21'], nodes['BR31'], nodes['AR12'], nodes['AR22'], nodes['AR32'], nodes['BL71'], nodes['BL81'], nodes['BL91'], nodes['AL72'], nodes['AL82'], nodes['AL92'], nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F']],

        nodes['19F']: [nodes['20F'], nodes['21F'], nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F'], nodes['CR11'], nodes['CR21'], nodes['CR31'], nodes['BR12'], nodes['BR22'], nodes['BR32'], nodes['CL71'], nodes['CL81'], nodes['CL91'], nodes['BL72'], nodes['BL82'], nodes['BL92'], nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B']],
        nodes['20F']: [nodes['19F'], nodes['21F'], nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F'], nodes['CR11'], nodes['CR21'], nodes['CR31'], nodes['BR12'], nodes['BR22'], nodes['BR32'], nodes['CL71'], nodes['CL81'], nodes['CL91'], nodes['BL72'], nodes['BL82'], nodes['BL92'], nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B']],
        nodes['21F']: [nodes['19F'], nodes['20F'], nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F'], nodes['CR11'], nodes['CR21'], nodes['CR31'], nodes['BR12'], nodes['BR22'], nodes['BR32'], nodes['CL71'], nodes['CL81'], nodes['CL91'], nodes['BL72'], nodes['BL82'], nodes['BL92'], nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B']],
        nodes['22F']: [nodes['19F'], nodes['20F'], nodes['21F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F'], nodes['CR11'], nodes['CR21'], nodes['CR31'], nodes['BR12'], nodes['BR22'], nodes['BR32'], nodes['CL71'], nodes['CL81'], nodes['CL91'], nodes['BL72'], nodes['BL82'], nodes['BL92'], nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B']],
        nodes['23F']: [nodes['19F'], nodes['20F'], nodes['21F'], nodes['22F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F'], nodes['CR11'], nodes['CR21'], nodes['CR31'], nodes['BR12'], nodes['BR22'], nodes['BR32'], nodes['CL71'], nodes['CL81'], nodes['CL91'], nodes['BL72'], nodes['BL82'], nodes['BL92'], nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B']],
        nodes['24F']: [nodes['19F'], nodes['20F'], nodes['21F'], nodes['22F'], nodes['23F'], nodes['25F'], nodes['26F'], nodes['27F'], nodes['CR11'], nodes['CR21'], nodes['CR31'], nodes['BR12'], nodes['BR22'], nodes['BR32'], nodes['CL71'], nodes['CL81'], nodes['CL91'], nodes['BL72'], nodes['BL82'], nodes['BL92'], nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B']],
        nodes['25F']: [nodes['19F'], nodes['20F'], nodes['21F'], nodes['22F'], nodes['23F'], nodes['24F'], nodes['26F'], nodes['27F'], nodes['CR11'], nodes['CR21'], nodes['CR31'], nodes['BR12'], nodes['BR22'], nodes['BR32'], nodes['CL71'], nodes['CL81'], nodes['CL91'], nodes['BL72'], nodes['BL82'], nodes['BL92'], nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B']],
        nodes['26F']: [nodes['19F'], nodes['20F'], nodes['21F'], nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['27F'], nodes['CR11'], nodes['CR21'], nodes['CR31'], nodes['BR12'], nodes['BR22'], nodes['BR32'], nodes['CL71'], nodes['CL81'], nodes['CL91'], nodes['BL72'], nodes['BL82'], nodes['BL92'], nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B']],
        nodes['27F']: [nodes['19F'], nodes['20F'], nodes['21F'], nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['CR11'], nodes['CR21'], nodes['CR31'], nodes['BR12'], nodes['BR22'], nodes['BR32'], nodes['CL71'], nodes['CL81'], nodes['CL91'], nodes['BL72'], nodes['BL82'], nodes['BL92'], nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B']],

        nodes['10B']: [nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B'], nodes['CR11'], nodes['CR21'], nodes['CR31'], nodes['BR12'], nodes['BR22'], nodes['BR32'], nodes['CL71'], nodes['CL81'], nodes['CL91'], nodes['BL72'], nodes['BL82'], nodes['BL92'], nodes['19F'], nodes['20F'], nodes['21F'], nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F']],
        nodes['11B']: [nodes['10B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B'], nodes['CR11'], nodes['CR21'], nodes['CR31'], nodes['BR12'], nodes['BR22'], nodes['BR32'], nodes['CL71'], nodes['CL81'], nodes['CL91'], nodes['BL72'], nodes['BL82'], nodes['BL92'], nodes['19F'], nodes['20F'], nodes['21F'], nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F']],
        nodes['12B']: [nodes['10B'], nodes['11B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B'], nodes['CR11'], nodes['CR21'], nodes['CR31'], nodes['BR12'], nodes['BR22'], nodes['BR32'], nodes['CL71'], nodes['CL81'], nodes['CL91'], nodes['BL72'], nodes['BL82'], nodes['BL92'], nodes['19F'], nodes['20F'], nodes['21F'], nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F']],
        nodes['13B']: [nodes['10B'], nodes['11B'], nodes['12B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B'], nodes['19F'], nodes['20F'], nodes['21F'], nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F'], nodes['CR11'], nodes['CR21'], nodes['CR31'], nodes['BR12'], nodes['BR22'], nodes['BR32'], nodes['CL71'], nodes['CL81'], nodes['CL91'], nodes['BL72'], nodes['BL82'], nodes['BL92']],
        nodes['14B']: [nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B'], nodes['19F'], nodes['20F'], nodes['21F'], nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F'], nodes['CR11'], nodes['CR21'], nodes['CR31'], nodes['BR12'], nodes['BR22'], nodes['BR32'], nodes['CL71'], nodes['CL81'], nodes['CL91'], nodes['BL72'], nodes['BL82'], nodes['BL92']],
        nodes['15B']: [nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['16B'], nodes['17B'], nodes['18B'], nodes['19F'], nodes['20F'], nodes['21F'], nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F'], nodes['CR11'], nodes['CR21'], nodes['CR31'], nodes['BR12'], nodes['BR22'], nodes['BR32'], nodes['CL71'], nodes['CL81'], nodes['CL91'], nodes['BL72'], nodes['BL82'], nodes['BL92']],
        nodes['16B']: [nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['17B'], nodes['18B'], nodes['CR11'], nodes['CR21'], nodes['CR31'], nodes['BR12'], nodes['BR22'], nodes['BR32'], nodes['CL71'], nodes['CL81'], nodes['CL91'], nodes['BL72'], nodes['BL82'], nodes['BL92'], nodes['19F'], nodes['20F'], nodes['21F'], nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F']],
        nodes['17B']: [nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['18B'], nodes['CR11'], nodes['CR21'], nodes['CR31'], nodes['BR12'], nodes['BR22'], nodes['BR32'], nodes['CL71'], nodes['CL81'], nodes['CL91'], nodes['BL72'], nodes['BL82'], nodes['BL92'], nodes['19F'], nodes['20F'], nodes['21F'], nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F']],
        nodes['18B']: [nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['CR11'], nodes['CR21'], nodes['CR31'], nodes['BR12'], nodes['BR22'], nodes['BR32'], nodes['CL71'], nodes['CL81'], nodes['CL91'], nodes['BL72'], nodes['BL82'], nodes['BL92'], nodes['19F'], nodes['20F'], nodes['21F'], nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F']],


        nodes['19B']: [nodes['20B'], nodes['21B'], nodes['22B'], nodes['23B'], nodes['24B'], nodes['25B'], nodes['26B'], nodes['27B'], nodes['CR12'], nodes['CR22'], nodes['CR32'],nodes['CL72'], nodes['CL82'], nodes['CL92']],
        nodes['20B']: [nodes['19B'], nodes['21B'], nodes['22B'], nodes['23B'], nodes['24B'], nodes['25B'], nodes['26B'], nodes['27B'], nodes['CR12'], nodes['CR22'], nodes['CR32'],nodes['CL72'], nodes['CL82'], nodes['CL92']],
        nodes['21B']: [nodes['19B'], nodes['20B'], nodes['22B'], nodes['23B'], nodes['24B'], nodes['25B'], nodes['26B'], nodes['27B'], nodes['CR12'], nodes['CR22'], nodes['CR32'],nodes['CL72'], nodes['CL82'], nodes['CL92']],
        nodes['22B']: [nodes['19B'], nodes['20B'], nodes['21B'], nodes['23B'], nodes['24B'], nodes['25B'], nodes['26B'], nodes['27B'], nodes['CR12'], nodes['CR22'], nodes['CR32'],nodes['CL72'], nodes['CL82'], nodes['CL92']],
        nodes['23B']: [nodes['19B'], nodes['20B'], nodes['21B'], nodes['22B'], nodes['24B'], nodes['25B'], nodes['26B'], nodes['27B'], nodes['CR12'], nodes['CR22'], nodes['CR32'],nodes['CL72'], nodes['CL82'], nodes['CL92']],
        nodes['24B']: [nodes['19B'], nodes['20B'], nodes['21B'], nodes['22B'], nodes['23B'], nodes['25B'], nodes['26B'], nodes['27B'], nodes['CR12'], nodes['CR22'], nodes['CR32'],nodes['CL72'], nodes['CL82'], nodes['CL92']],
        nodes['25B']: [nodes['19B'], nodes['20B'], nodes['21B'], nodes['22B'], nodes['23B'], nodes['24B'], nodes['26B'], nodes['27B'], nodes['CL72'], nodes['CL82'], nodes['CL92'], nodes['CR12'], nodes['CR22'], nodes['CR32']],
        nodes['26B']: [nodes['19B'], nodes['20B'], nodes['21B'], nodes['22B'], nodes['23B'], nodes['24B'], nodes['25B'], nodes['27B'], nodes['CL72'], nodes['CL82'], nodes['CL92'], nodes['CR12'], nodes['CR22'], nodes['CR32']],
        nodes['27B']: [nodes['19B'], nodes['20B'], nodes['21B'], nodes['22B'], nodes['23B'], nodes['24B'], nodes['25B'], nodes['26B'], nodes['CL72'], nodes['CL82'], nodes['CL92'], nodes['CR12'], nodes['CR22'], nodes['CR32']],

        nodes['AR11']: [nodes['AR21'], nodes['AR31'], nodes['1F'], nodes['2F'], nodes['3F'], nodes['4F'], nodes['5F'], nodes['6F'], nodes['7F'], nodes['8F'], nodes['9F'],nodes['START'], nodes['AR13'], nodes['AR23'], nodes['AR33']], #,nodes['BR12'], nodes['BR22'], nodes['BR32']
        nodes['AR21']: [nodes['AR11'], nodes['AR31'], nodes['1F'], nodes['2F'], nodes['3F'], nodes['4F'], nodes['5F'], nodes['6F'], nodes['7F'], nodes['8F'], nodes['9F'],nodes['START'], nodes['AR13'], nodes['AR23'], nodes['AR33']],
        nodes['AR31']: [nodes['AR11'], nodes['AR21'], nodes['1F'], nodes['2F'], nodes['3F'], nodes['4F'], nodes['5F'], nodes['6F'], nodes['7F'], nodes['8F'], nodes['9F'],nodes['START'], nodes['AR13'], nodes['AR23'], nodes['AR33']],
        nodes['AL71']: [nodes['AL81'], nodes['AL91'], nodes['1F'], nodes['2F'], nodes['3F'], nodes['4F'], nodes['5F'], nodes['6F'], nodes['7F'], nodes['8F'], nodes['9F'], nodes['AL73'], nodes['AL83'], nodes['AL93']],
        nodes['AL81']: [nodes['AL71'], nodes['AL91'], nodes['1F'], nodes['2F'], nodes['3F'], nodes['4F'], nodes['5F'], nodes['6F'], nodes['7F'], nodes['8F'], nodes['9F'], nodes['AL73'], nodes['AL83'], nodes['AL93']],
        nodes['AL91']: [nodes['AL71'], nodes['AL81'], nodes['1F'], nodes['2F'], nodes['3F'], nodes['4F'], nodes['5F'], nodes['6F'], nodes['7F'], nodes['8F'], nodes['9F'], nodes['AL73'], nodes['AL83'], nodes['AL93']], #,nodes['BL71'], nodes['BL81'], nodes['BL91']

        nodes['BR11']: [nodes['BR21'], nodes['BR31'], nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F'], nodes['AR12'], nodes['AR22'], nodes['AR32'],nodes['1B'], nodes['2B'], nodes['3B'], nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B'], nodes['BR13'], nodes['BR23'], nodes['BR33'], nodes['AR13'], nodes['AR23'], nodes['AR33']],
        nodes['BR21']: [nodes['BR11'], nodes['BR31'], nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F'], nodes['AR12'], nodes['AR22'], nodes['AR32'],nodes['1B'], nodes['2B'], nodes['3B'], nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B'], nodes['BR13'], nodes['BR23'], nodes['BR33'], nodes['AR13'], nodes['AR23'], nodes['AR33']],
        nodes['BR31']: [nodes['BR11'], nodes['BR21'], nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F'], nodes['AR12'], nodes['AR22'], nodes['AR32'],nodes['1B'], nodes['2B'], nodes['3B'], nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B'], nodes['BR13'], nodes['BR23'], nodes['BR33'], nodes['AR13'], nodes['AR23'], nodes['AR33']],
        nodes['BL71']: [nodes['BL81'], nodes['BL91'], nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F'], nodes['AL72'], nodes['AL82'], nodes['AL92'],nodes['1B'], nodes['2B'], nodes['3B'], nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B'], nodes['BL73'], nodes['BL83'], nodes['BL93'], nodes['AL73'], nodes['AL83'], nodes['AL93']],
        nodes['BL81']: [nodes['BL71'], nodes['BL91'], nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F'], nodes['AL72'], nodes['AL82'], nodes['AL92'],nodes['1B'], nodes['2B'], nodes['3B'], nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B'], nodes['BL73'], nodes['BL83'], nodes['BL93'], nodes['AL73'], nodes['AL83'], nodes['AL93']],
        nodes['BL91']: [nodes['BL71'], nodes['BL81'], nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F'], nodes['AL72'], nodes['AL82'], nodes['AL92'],nodes['1B'], nodes['2B'], nodes['3B'], nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B'], nodes['BL73'], nodes['BL83'], nodes['BL93'], nodes['AL73'], nodes['AL83'], nodes['AL93']],

        nodes['CR11']: [nodes['CR21'], nodes['CR31'], nodes['19F'], nodes['20F'], nodes['21F'], nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F'], nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B'], nodes['CR13'], nodes['CR23'], nodes['CR33'], nodes['BR13'], nodes['BR23'], nodes['BR33']],
        nodes['CR21']: [nodes['CR11'], nodes['CR31'], nodes['19F'], nodes['20F'], nodes['21F'], nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F'], nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B'], nodes['CR13'], nodes['CR23'], nodes['CR33'], nodes['BR13'], nodes['BR23'], nodes['BR33']],
        nodes['CR31']: [nodes['CR11'], nodes['CR21'], nodes['19F'], nodes['20F'], nodes['21F'], nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F'], nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B'], nodes['CR13'], nodes['CR23'], nodes['CR33'], nodes['BR13'], nodes['BR23'], nodes['BR33']],
        nodes['CL71']: [nodes['CL81'], nodes['CL91'], nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B'], nodes['19F'], nodes['20F'], nodes['21F'], nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F'], nodes['CL73'], nodes['CL83'], nodes['CL93'], nodes['BL73'], nodes['BL83'], nodes['BL93']],
        nodes['CL81']: [nodes['CL71'], nodes['CL91'], nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B'], nodes['19F'], nodes['20F'], nodes['21F'], nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F'], nodes['CL73'], nodes['CL83'], nodes['CL93'], nodes['BL73'], nodes['BL83'], nodes['BL93']],
        nodes['CL91']: [nodes['CL71'], nodes['CL81'], nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B'], nodes['19F'], nodes['20F'], nodes['21F'], nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F'], nodes['CL73'], nodes['CL83'], nodes['CL93'], nodes['BL73'], nodes['BL83'], nodes['BL93']],
        
        nodes['AR12']: [nodes['1B'], nodes['2B'], nodes['3B'],nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B'],nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F'],nodes['BR11'], nodes['BR21'], nodes['BR31'], nodes['BR13'], nodes['BR23'], nodes['BR33']],
        nodes['AR22']: [nodes['1B'], nodes['2B'], nodes['3B'],nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B'],nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F'],nodes['BR11'], nodes['BR21'], nodes['BR31'], nodes['BR13'], nodes['BR23'], nodes['BR33']],
        nodes['AR32']: [nodes['1B'], nodes['2B'], nodes['3B'],nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B'],nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F'],nodes['BR11'], nodes['BR21'], nodes['BR31'], nodes['BR13'], nodes['BR23'], nodes['BR33']],
        nodes['AL72']: [nodes['1B'], nodes['8B'], nodes['3B'],nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B'],nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F'],nodes['BL71'], nodes['BL81'], nodes['BL91'], nodes['BL73'], nodes['BL83'], nodes['BL93']],
        nodes['AL82']: [nodes['1B'], nodes['2B'], nodes['3B'],nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B'],nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F'],nodes['BL71'], nodes['BL81'], nodes['BL91'], nodes['BL73'], nodes['BL83'], nodes['BL93']],
        nodes['AL92']: [nodes['1B'], nodes['8B'], nodes['3B'],nodes['4B'], nodes['5B'], nodes['6B'], nodes['7B'], nodes['8B'], nodes['9B'],nodes['10F'], nodes['11F'], nodes['12F'], nodes['13F'], nodes['14F'], nodes['15F'], nodes['16F'], nodes['17F'], nodes['18F'],nodes['BL71'], nodes['BL81'], nodes['BL91'], nodes['BL73'] ,nodes['BL83'], nodes['BL93']],

        nodes['BR12']: [nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B'],nodes['19F'], nodes['20F'], nodes['21F'],nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F'], nodes['CR13'], nodes['CR23'], nodes['CR33']],
        nodes['BR22']: [nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B'],nodes['19F'], nodes['20F'], nodes['21F'],nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F'], nodes['CR13'], nodes['CR23'], nodes['CR33']],
        nodes['BR32']: [nodes['10B'], nodes['11B'], nodes['12B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B'],nodes['19F'], nodes['20F'], nodes['21F'],nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F'], nodes['CR13'], nodes['CR23'], nodes['CR33']],
        nodes['BL72']: [nodes['16B'], nodes['17B'], nodes['18B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B'],nodes['19F'], nodes['20F'], nodes['21F'],nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F'], nodes['CL73'], nodes['CL83'], nodes['CL93']],
        nodes['BL82']: [nodes['16B'], nodes['17B'], nodes['18B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B'],nodes['19F'], nodes['20F'], nodes['21F'],nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F'], nodes['CL73'], nodes['CL83'], nodes['CL93']],
        nodes['BL92']: [nodes['16B'], nodes['17B'], nodes['18B'], nodes['13B'], nodes['14B'], nodes['15B'], nodes['16B'], nodes['17B'], nodes['18B'],nodes['19F'], nodes['20F'], nodes['21F'],nodes['22F'], nodes['23F'], nodes['24F'], nodes['25F'], nodes['26F'], nodes['27F'], nodes['CL73'], nodes['CL83'], nodes['CL93']],
        
        nodes['CR12']: [nodes['19B'], nodes['20B'], nodes['21B'], nodes['22B'], nodes['23B'], nodes['24B'], nodes['25B'], nodes['26B'], nodes['27B'], nodes['CR13'], nodes['CR23'], nodes['CR33']],
        nodes['CR22']: [nodes['19B'], nodes['20B'], nodes['21B'], nodes['22B'], nodes['23B'], nodes['24B'], nodes['25B'], nodes['26B'], nodes['27B'], nodes['CR13'], nodes['CR23'], nodes['CR33']],
        nodes['CR32']: [nodes['19B'], nodes['20B'], nodes['21B'], nodes['22B'], nodes['23B'], nodes['24B'], nodes['25B'], nodes['26B'], nodes['27B'], nodes['CR13'], nodes['CR23'], nodes['CR33']],
        nodes['CL72']: [nodes['25B'], nodes['26B'], nodes['27B'], nodes['22B'], nodes['23B'], nodes['24B'], nodes['25B'], nodes['26B'], nodes['27B'], nodes['CL73'], nodes['CL83'], nodes['CL93']],
        nodes['CL82']: [nodes['25B'], nodes['26B'], nodes['27B'], nodes['22B'], nodes['23B'], nodes['24B'], nodes['25B'], nodes['26B'], nodes['27B'], nodes['CL73'], nodes['CL83'], nodes['CL93']],
        nodes['CL92']: [nodes['25B'], nodes['26B'], nodes['27B'], nodes['22B'], nodes['23B'], nodes['24B'], nodes['25B'], nodes['26B'], nodes['27B'], nodes['CL73'], nodes['CL83'], nodes['CL93']],

        nodes['AR13']: [nodes['AR12'],nodes['AR22'],nodes['AR32'],nodes['BR13'],nodes['START']],
        nodes['AR23']: [nodes['AR12'],nodes['AR22'],nodes['AR32'],nodes['BR23'],nodes['START']],
        nodes['AR33']: [nodes['AR12'],nodes['AR22'],nodes['AR32'],nodes['BR33'],nodes['START']],
        nodes['BR13']: [nodes['BR12'],nodes['BR22'],nodes['BR32'],nodes['CR13'],nodes['AR13']],
        nodes['BR23']: [nodes['BR12'],nodes['BR22'],nodes['BR32'],nodes['CR23'],nodes['AR23']],
        nodes['BR33']: [nodes['BR12'],nodes['BR22'],nodes['BR32'],nodes['CR33'],nodes['AR33']],
        nodes['CR13']: [nodes['CR12'],nodes['CR22'],nodes['CR32'],nodes['AR13']],
        nodes['CR23']: [nodes['CR12'],nodes['CR22'],nodes['CR32'],nodes['AR23']],
        nodes['CR33']: [nodes['CR12'],nodes['CR22'],nodes['CR32'],nodes['AR33']],
        nodes['AL73']: [nodes['AL72'],nodes['AL82'],nodes['AL93'],nodes['BL73']],
        nodes['AL83']: [nodes['AL72'],nodes['AL82'],nodes['AL93'],nodes['BL83']],
        nodes['AL93']: [nodes['AL72'],nodes['AL82'],nodes['AL93'],nodes['BL93']],
        nodes['BL73']: [nodes['BL72'],nodes['BL82'],nodes['BL93'],nodes['CL73'],nodes['AL73']],
        nodes['BL83']: [nodes['BL72'],nodes['BL82'],nodes['BL93'],nodes['CL83'],nodes['AL83']],
        nodes['BL93']: [nodes['BL72'],nodes['BL82'],nodes['BL93'],nodes['CL93'],nodes['AL93']],
        nodes['CL73']: [nodes['CL72'],nodes['CL82'],nodes['CL93'],nodes['AL73']],
        nodes['CL83']: [nodes['CL72'],nodes['CL82'],nodes['CL93'],nodes['AL83']],
        nodes['CL93']: [nodes['CL72'],nodes['CL82'],nodes['CL93'],nodes['AL93']]
    }
        
    start = nodes[st]
    goal = nodes[nd]
    path = a_star_3d(start, goal, 10, 10, 10)
    output = []
    if path:
        for step in path:
         #   print(step.x, step.y, step.z,step.yaw,step.name)
            output.append([step.x,step.y,step.z,step.yaw,step.name])
    else:
      #  print("No path found")
        pass
    return output

def distance2d(node,goal):
    return ((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2 + (node[2] - goal[2]) ** 2) ** 0.5

def wayp(beds):
    # getting beds from waypoint file
    # creating a list of waypoints from beds
    wp_front=[]
    wp_back=[] 
    for a in beds:
        wp_front.append(str(a)+"F")
        wp_back.append(str(a)+"B")
    total_path = []
    
   # print(wp_front,wp_back)
        
    # sorting wp on the basis of bed number
    wp_front_sorted = sorted(wp_front, key=lambda x: int(x[:-1]))
    wp_back_sorted = sorted(wp_back, key=lambda x: int(x[:-1]))

   # print(wp_front_sorted)
   # print(wp_back_sorted)
    
    empt = []
    final_path = []
    last_coord=[]
    # front 1-9
    
    total_path.extend(best_path([x for x in wp_front_sorted if int(x[:-1]) <= 9]))
    final_path.extend(total_path)
    
    total_path = []
    total_path.extend(best_path([x for x in wp_back_sorted if int(x[:-1]) <= 9]+[x for x in wp_front_sorted if int(x[:-1]) >= 10 and int(x[:-1]) <= 18]))
    final_path.extend(total_path)
    
    total_path = []
    total_path.extend(best_path([x for x in wp_back_sorted if int(x[:-1]) >= 10 and int(x[:-1]) <= 18]+[x for x in wp_front_sorted if int(x[:-1]) >= 19 and int(x[:-1]) <= 27]))
    final_path.extend(total_path)
    
    total_path = []
    total_path.extend(best_path([x for x in wp_back_sorted if int(x[:-1]) >= 19 and int(x[:-1]) <= 27]))
    final_path.extend(total_path)
    
 #   print("end to start")
    total_path = []
    total_path.extend(pathfinder(start,'START'))
    final_path.extend(total_path)
    
   # print(final_path)
    return final_path


def best_path(wp):
    print(wp)

    if wp==[]:
        return wp
    
    global path_value
    global start
    
    
    f_initial = float('inf')
    f_path = []
    f_point = None
    path_final = []
    
    
    while wp  !=  []:
        for point in range(len(wp)):
            path = []
            
            path_value = 0
            j = pathfinder(start,wp[point])
           # print(path_value)
            if path_value<f_initial:
                f_path = j
                f_point = wp[point]
                f_initial = path_value
              #  print(path_value)
                
        f_initial = float('inf')
     #   print(wp)
      #  print(f_point)
        wp.remove(f_point)
        path_final.extend(f_path)
        start = f_point

   
        
  
    return path_final
