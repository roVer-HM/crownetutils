class Building(object):
    def __init__(self, floors):
        self._floors = [None] * floors

    def set_floor_data(self, floor_number, data):
        self._floors[floor_number] = data

    def get_floor_data(self, floor_number):
        return self._floors[floor_number]


class BuildingItem(object):
    def __init__(self, floors):
        self._floors = [None] * floors

    def __setitem__(self, floor_number, data):
        self._floors[floor_number] = data

    def __getitem__(self, floor_number):
        return self._floors[floor_number]


if __name__ == "__main__":
    building1 = Building(4)  # Construct a building with 4 floors
    building1.set_floor_data(0, "Reception")
    building1.set_floor_data(1, "ABC Corp")
    building1.set_floor_data(2, "DEF Inc")
    print(building1.get_floor_data(2))

    building_item = BuildingItem(4)
    building_item[0] = "Reception"
    building_item[1] = "ABC Corp"
    building_item[2] = "DEF Inc"
    print(building_item[0:2])
