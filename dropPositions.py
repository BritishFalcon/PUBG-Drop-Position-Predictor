# Import libraries to API calls
import requests
import json
import matplotlib.pyplot as plt
from functions import *

def getDropPositions(match_id, platform, apiKey):

    url = f"https://api.pubg.com/shards/{platform}/matches/{match_id}"
    response = getCall(url, apiKey)

    for items in response["included"]:
        if items["type"] == "asset":
            telemetryURL = items["attributes"]["URL"]

    # Download telemetry file
    telemetry = requests.get(telemetryURL)
    telemetry = telemetry.json()

    positions = []
    checked_names = []

    playerCount = 0
    for items in telemetry:

        # Find LogParachuteLanding event
        if items["_T"] == "LogParachuteLanding":

            # Check if the name has already been checked
            if items["character"]["name"] not in checked_names:
                playerCount += 1
                # Record the x and y coordinates
                x = items["character"]["location"]["x"] / 100
                y = 8000 - items["character"]["location"]["y"] / 100
                positions.append([x, y])

                # Add the name to the checkedNames list
                checked_names.append(items["character"]["name"])

    aircraft_start_x = None
    aircraft_start_y = None
    # Find the start and end position of the dropping plane
    for items in telemetry:
        if items["_T"] == "LogPlayerPosition":
            if items["vehicle"] != None:
                if items["character"]["location"]["x"] >= 0 and items["character"]["location"]["y"] >= 0:
                    if aircraft_start_x is None and aircraft_start_y is None:
                        aircraft_start_x = items["character"]["location"]["x"] / 100
                        aircraft_start_y = 8000 - items["character"]["location"]["y"] / 100
                    elif aircraft_start_x != items["character"]["location"]["x"] / 100 and aircraft_start_y != 8000 - items["character"]["location"]["y"] / 100: # If x is already set, then this is the second position
                        x2 = items["character"]["location"]["x"] / 100
                        y2 = 8000 - items["character"]["location"]["y"] / 100

                        # Calculate the slope of the line
                        m = (y2 - aircraft_start_y) / (x2 - aircraft_start_x)
                        # Calculate the y-intercept
                        c = aircraft_start_y - m * aircraft_start_x

                        if 0 <= aircraft_start_y <= 8000: # West or east side
                            if aircraft_start_x <= 4000: # West
                                aircraft_start_x = 0
                                aircraft_end_x = 8000
                                aircraft_end_y = m * aircraft_end_x + c
                            else: # East
                                aircraft_start_x = 8000
                                aircraft_end_x = 0
                                aircraft_end_y = m * aircraft_end_x + c

                        else: # North or south side
                            if aircraft_start_y <= 4000: # South
                                aircraft_start_y = 0
                                aircraft_end_y = 8000
                                aircraft_end_x = (aircraft_end_y - c) / m
                            else: # North
                                aircraft_start_y = 8000
                                aircraft_end_y = 0
                                aircraft_end_x = (aircraft_end_y - c) / m

                        break

    # Use quadrat sampling for points on the map
    sampleArray = [[0] * 80 for i in range(80)]
    for x in range(len(sampleArray)):
        for y in range(len(sampleArray)):

            # Count the number of points in each quadrant
            count = 0
            for i in positions:
                if i[0] >= x * 100 and i[0] < (x + 1) * 100:
                    if i[1] >= y * 100 and i[1] < (y + 1) * 100:
                        count += 1

            sampleArray[y][x] = count / playerCount

    return [aircraft_start_x, aircraft_start_y, aircraft_end_x, aircraft_end_y], sampleArray

if __name__ == "__main__":
    import csv
    import numpy
    import itertools
    # Get the drop positions
    apiKey = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJqdGkiOiIxYTY1NDUzMC02MjBmLTAxM2ItNGRiOS0wMzFjNzRhOWU0NTciLCJpc3MiOiJnYW1lbG9ja2VyIiwiaWF0IjoxNjcxNDg0MDYzLCJwdWIiOiJibHVlaG9sZSIsInRpdGxlIjoicHViZyIsImFwcCI6ImJyaXRpc2hmYWxjb25nIn0.4vwHivcvKIG_XWyCvXfgilPDyDItaHE7ZhsahPdgf3I"
    platform = "pc-eu"
    match_id = "52a5331a-6290-48ff-8849-015ab3879b8a"

    response = getCall("https://api.pubg.com/shards/steam/samples", apiKey)

    matches_sample = response["data"]["relationships"]["matches"]["data"]

    # Open a file for writing in the CSV format
    with open('games.csv', 'w', newline='') as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)

        # Write the column names as the first row
        writer.writerow(['start_x', 'start_y', 'end_x', 'end_y'] + [f'quadrant_{i}' for i in range(6400)])

        count = 0
        # Iterate through the matches
        for match in matches_sample:
            count += 1
            # Get the drop positions
            response = getCall(f"https://api.pubg.com/shards/pc-eu/matches/{match['id']}", apiKey)
            # Check if the match is on Erangel
            if response["data"]["attributes"]["mapName"] == "Baltic_Main":
                x, y = getDropPositions(match["id"], platform, apiKey)

                # Flatten the output data into a single list
                y = numpy.array(y).flatten().tolist()

                writer.writerow(x + y)

                print(f"Finished {count} matches")