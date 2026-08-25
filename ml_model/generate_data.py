import random
import csv
zones = ["A", "B", "C", "D"]

# hours (24-hour format) when each zone tends to get busy
busy_hours = {
    "A": [8, 9, 10, 11],       # morning rush
    "B": [17, 18, 19, 20],     # evening rush
    "C": [],                   # stays steady/low
    "D": [],                   # stays steady/low
}
def generate_order():
    hour = random.randint(0, 23)

    # find which zones are "busy" at this hour
    busy_zones = [zone for zone, hours in busy_hours.items() if hour in hours]

    if busy_zones:
        # 70% chance the order comes from the busy zone at this hour
        if random.random() < 0.7:
            zone = random.choice(busy_zones)
        else:
            zone = random.choice(zones)
    else:
        # no zone is "busy" at this hour, pick randomly
        zone = random.choice(zones)

    return hour, zone
def generate_dataset(num_orders=1000, filename="warehouse_orders.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["hour", "zone"])  # header row

        for _ in range(num_orders):
            hour, zone = generate_order()
            writer.writerow([hour, zone])

    print(f"Generated {num_orders} orders and saved to {filename}")

# Run the generator
generate_dataset()