from fast_flights import FlightData, Passengers, create_filter


def generate_url_one_way(
    origin,
    destination,
    depart_date,
    passenger,
    seat_type="economy",
    currency="USD",
):
    # Create a new filter
    filter = create_filter(
        flight_data=[
            FlightData(
                date=depart_date,  # Date of departure for outbound flight
                from_airport=origin,
                to_airport=destination,
            ),
        ],
        trip="one-way",  # Trip (round-trip, one-way)
        seat=seat_type,  # Seat (economy, premium-economy, business or first)
        passengers=passenger,
    )

    # Encode the filter to base64
    b64 = filter.as_b64().decode("utf-8")

    # Construct the Google Flights URL with currency set to USD
    url = f"https://www.google.com/travel/flights?tfs={b64}&curr={currency}"
    return url


def generate_url_round_trip(
    origin,
    destination,
    depart_date,
    return_date,
    passenger,
    seat_type="economy",
    currency="USD",
):
    # Create a new filter
    filter = create_filter(
        flight_data=[
            FlightData(
                date=depart_date,  # Date of departure for outbound flight
                from_airport=origin,
                to_airport=destination,
            ),
            FlightData(
                date=return_date,  # Date of departure for return flight
                from_airport=destination,
                to_airport=origin,
            ),
        ],
        trip="round-trip",  # Trip (round-trip, one-way)
        seat=seat_type,  # Seat (economy, premium-economy, business or first)
        passengers=passenger,
    )

    # Encode the filter to base64
    b64 = filter.as_b64().decode("utf-8")

    # Construct the Google Flights URL with currency set to USD
    url = f"https://www.google.com/travel/flights?tfs={b64}&curr={currency}"
    return url
