import setting

# Define settings for different maps
MAP_SETTINGS = {
    'map_house.json': {
        'states': ['Living Room', 'Kitchen', 'Office', 'Hallway', 'Dining Room'],
        'cpt': setting.COND_PROB_TABLE_HOUSE,
        'coord_state': setting.COORD_STATE_HOUSE
    },
    'map_office.json': {
        'states': ['Conference Room', 'Suite A', 'Hallway', 'Break Room', 'Lab B', 'Suite B', 'Room C', 'Lab C', 'Inventory'],
        'cpt': setting.COND_PROB_TABLE_OFFICE,
        'coord_state': setting.COORD_STATE_OFFICE
    }
}