import math
import matplotlib.pyplot as plt

class AStarPlanner:
    def __init__(self, ox, oy, resolution, rr, w1=1.0, w2=1.0, w3=1.0):
        """
        Initialisierung des A*-Planers mit mehreren Zielparametern
        :param ox: Liste der x-Positionen der Hindernisse [m]
        :param oy: Liste der y-Positionen der Hindernisse [m]
        :param resolution: Auflösung des Gitters [m]
        :param rr: Radius des Roboters [m]
        :param w1: Gewichtung für die Pfadlänge
        :param w2: Gewichtung für den Abstand zu Hindernissen (Clearance)
        :param w3: Gewichtung für Energiekosten (z. B. Wendekosten)
        """
        self.resolution = resolution
        self.rr = rr
        self.ox = ox
        self.oy = oy
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.motion = self.get_motion_model()
        self.calc_obstacle_map()

    class Node:
        def __init__(self, x, y, cost, parent_index):
            """
            Definiert einen Knoten im Suchbaum
            :param x: x-Index des Knotens im Gitter
            :param y: y-Index des Knotens im Gitter
            :param cost: Gesamtkosten bis zu diesem Knoten
            :param parent_index: Index des Elternknotens
            """
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index

    def planning(self, sx, sy, gx, gy):
        """
        Durchführung der A*-Pfadsuche
        :param sx: Startposition x [m]
        :param sy: Startposition y [m]
        :param gx: Zielposition x [m]
        :param gy: Zielposition y [m]
        :return: rx, ry - Listen mit den x- und y-Positionen des geplanten Pfads
        """
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if not open_set:
                print("Open set is empty")
                break

            # Auswahl des Knotens mit den geringsten Kosten
            # c_id = min(open_set, key=lambda o: open_set[o].cost  +
            #            self.calc_heuristic(goal_node, open_set[o])) # Index finden
            
            # Auswahl des Knotens mit den geringsten Kosten mit einbeziehung der Gewichtungen
            c_id = min(open_set, key=lambda o:                          # Index finden
                       open_set[o].cost * self.w1 +                     # Pfadlänge mit gewichtung
                       self.calc_heuristic(goal_node, open_set[o]) +    # Heuristik
                       self.w2 / self.calc_clearance(node) +            # Abstand zu Hindernissen mit gewichtung
                       self.w3 * self.calc_energy_cost(current, node))  # Energiekosten mit gewichtung
## Herausfinden welche art und weise der implemntierung am sinnvollsten ist, die energiekosten können nur durch den Vorherigen Punkt mit in die Kosten eingrechnet werden
# Die Kosten für den abstand zu den Hindernissen können nur durch den aktuellen Punkt berechnet werden, da der Abstand zu den Hindernissen nur durch den aktuellen Punkt berechnet werden kann.
# Heißt diese Teile müssten doch bereits bei der berechnung der Kosten für g(n) mit hinz8 gezogen werden und auch gewichtet werden.
# Damit sollte bei der AUswahl des Knotens mit den ginsgten Kosten nur die Heuristit stehen. Kann auch die Heuristk für jeden Punkt direkt berechnet werden? dann muss nur der mindest wert Abgefragt werden.
# Ist da effizenter da nicht jeder wert imme wieder neu berechnet werden muss.                        
            
            current = open_set[c_id]                                # Knoten speichern         

            # Animation der aktuellen Suche
            if True:
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                # Pause für die Animation
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            # Überprüfe, ob das Ziel erreicht wurde
            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            del open_set[c_id]
            closed_set[c_id] = current

            # Erweitere die Nachbarknoten
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue
                
                # ** Auskommentiert da hier nur die Pfadlänge betrachtet wird
                # für die Bewechnung der Kosten eines Punktes muss zu einem anden Zeitpunktpassieren.
                # # Multi-Objective-Kostenberechnung
                # g_cost = current.cost + self.motion[i][2]
                # h_cost = self.calc_heuristic(goal_node, node)
                # clearance_cost = self.w2 / self.calc_clearance(node)
                # energy_cost = self.w3 * self.calc_energy_cost(current, node)

                #node.cost = g_cost + h_cost + self.w1 * g_cost + clearance_cost + energy_cost

                if n_id not in open_set:   
                    open_set[n_id] = node  # discovered a new node
                else:                      # Knoten bereits in der offenen Liste
                    if open_set[n_id].cost > node.cost:  # Wenn Kosten geringer wird der Knoten aktualisiert
                        # This path is the best until now. record it
                        open_set[n_id] = node  # Update node

        rx, ry = self.calc_final_path(goal_node, closed_set)
        return rx, ry


    def calc_energy_cost(self, current, neighbor):
        """ Berechnet die Energiekosten basierend auf Wendewinkeln """
        # Berechnung der Differenzen der Koordinaten
        dx1, dy1 = current.x - neighbor.x, current.y - neighbor.y
        dx2, dy2 = neighbor.x - current.x, neighbor.y - current.y
        
        # Berechnung des absoluten Unterschieds der Winkel
        angle = abs(math.atan2(dy2, dx2) - math.atan2(dy1, dx1))
        
        # Rückgabe des Winkels als Energiekosten
        return angle

    def calc_clearance(self, node):
        """ Berechnet den minimalen Abstand eines Knotens zu Hindernissen """
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)
        min_dist = float("inf")
        for ox, oy in zip(self.ox, self.oy):
            d = math.hypot(px - ox, py - oy)
            if d < min_dist:
                min_dist = d
        return min_dist

    def calc_final_path(self, goal_node, closed_set):
        """ Rekonstruiert den finalen Pfad vom Start- zum Zielknoten """
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)],[
                  self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index
        return rx[::-1], ry[::-1]

    def calc_heuristic(self, n1, n2):
        """ Berechnet die heuristischen Kosten zwischen zwei Knoten """
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    def calc_grid_position(self, index, min_pos):
        """ Konvertiert Gitterindex in eine physikalische Position """
        return index * self.resolution + min_pos

    def calc_xy_index(self, position, min_pos):
        """ Konvertiert eine Position in einen Gitterindex """
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        """ Berechnet den linearen Index eines Knotens im Gitter """
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        """ Überprüft, ob ein Knoten gültig ist (nicht außerhalb des Gitters) """
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x or py < self.min_y or px >= self.max_x or py >= self.max_y:
            return False

        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self):
        """ Erstellt die Hinderniskarte basierend auf Hindernispositionen """
        self.min_x = round(min(self.ox))
        self.min_y = round(min(self.oy))
        self.max_x = round(max(self.ox))
        self.max_y = round(max(self.oy))
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

        self.obstacle_map = [[False for _ in range(self.y_width)] for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for ox, oy in zip(self.ox, self.oy):
                    if math.hypot(ox - x, oy - y) <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        """ Definiert Bewegungsmöglichkeiten: Geradeaus und diagonal mit Kosten """
        return [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
                [-1, -1, math.sqrt(2)], [-1, 1, math.sqrt(2)],
                [1, -1, math.sqrt(2)], [1, 1, math.sqrt(2)]]

def main():
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    sx, sy = 10.0, 10.0
    gx, gy = 50.0, 50.0
    grid_size = 2.0
    robot_radius = 1.0

    plt.plot(ox, oy, ".k")
    plt.plot(sx, sy, "og")
    plt.plot(gx, gy, "xb")
    plt.grid(True)
    plt.axis("equal")

    planner = AStarPlanner(ox, oy, grid_size, robot_radius, w1=1.0, w2=0.1, w3=0)
    rx, ry = planner.planning(sx, sy, gx, gy)

    plt.plot(ox, oy, ".k")
    plt.plot(sx, sy, "og")
    plt.plot(gx, gy, "xb")
    plt.plot(rx, ry, "-r")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

if __name__ == '__main__':
    main()
