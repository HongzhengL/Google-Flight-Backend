from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Literal, Optional, List
import logging
import os
from fast_flights import FlightData, get_flights, Passengers

from app.url import generate_url_one_way, generate_url_round_trip

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key configuration
API_KEY = os.getenv("API_KEY", "your-secret-api-key-here")
security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

app = FastAPI(title="Flight Search API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FlightSearchRequest(BaseModel):
    origin: str
    destination: str
    departure_date: str
    return_date: Optional[str] = None
    currency: Optional[str] = "USD"
    seat_type: Literal["economy", "premium-economy", "business", "first"] = "economy"
    adults: int = 1
    children: int = 0
    infants_in_seat: int = 0
    infants_on_lap: int = 0


class FlightInfo(BaseModel):
    airline: str
    departure_time: str
    arrival_time: str
    duration: str
    price: str
    currency: str
    stops: int
    booking_url: Optional[str] = None


class FlightSearchResponse(BaseModel):
    success: bool
    flights: List[FlightInfo]
    total_results: int
    error: Optional[str] = None


@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Flight API is running"}


@app.post("/search-flights", response_model=FlightSearchResponse)
async def search_flights(request: FlightSearchRequest, api_key: str = Depends(verify_api_key)):
    try:
        logger.info(f"Searching flights from {request.origin} to {request.destination}")

        # Validate required fields
        if not request.origin or not request.destination or not request.departure_date:
            raise HTTPException(
                status_code=400,
                detail="Missing required parameters: origin, destination, departure_date",
            )

        is_round_trip = request.return_date is not None

        # Create flight data object
        flight_data = [
            FlightData(
                date=request.departure_date,
                from_airport=request.origin.upper(),
                to_airport=request.destination.upper(),
            )
        ]

        if is_round_trip:
            assert request.return_date is not None  # suppress warning
            flight_data.append(
                FlightData(
                    date=request.return_date,
                    from_airport=request.destination.upper(),
                    to_airport=request.origin.upper(),
                )
            )

        # Create passengers object
        passengers = Passengers(
            adults=request.adults,
            children=request.children,
            infants_in_seat=request.infants_in_seat,
            infants_on_lap=request.infants_on_lap,
        )

        # Search for flights
        logger.info("Calling fast_flights API...")
        result = get_flights(
            flight_data=flight_data,
            trip="round-trip" if is_round_trip else "one-way",
            passengers=passengers,
            seat=request.seat_type,
        )

        # Convert flights to response format
        flight_results = []
        assert request.currency is not None
        if not is_round_trip:
            for flight in result.flights:
                flight_info = FlightInfo(
                    airline=flight.name,
                    departure_time=flight.departure,
                    arrival_time=flight.arrival,
                    duration=flight.duration,
                    price=flight.price,
                    currency="USD",
                    stops=flight.stops,
                    booking_url=generate_url_one_way(
                        origin=request.origin,
                        destination=request.destination,
                        depart_date=request.departure_date,
                        passenger=passengers,
                        seat_type=request.seat_type,
                        currency=request.currency,
                    ),
                )
                flight_results.append(flight_info)
        elif is_round_trip:
            for flight in result.flights:
                flight_info = FlightInfo(
                    airline=flight.name,
                    departure_time=flight.departure,
                    arrival_time=flight.arrival,
                    duration=flight.duration,
                    price=flight.price,
                    currency="USD",
                    stops=flight.stops,
                    booking_url=generate_url_round_trip(
                        origin=request.origin,
                        destination=request.destination,
                        depart_date=request.departure_date,
                        return_date=request.return_date,
                        passenger=passengers,
                        seat_type=request.seat_type,
                        currency=request.currency,
                    ),
                )
                flight_results.append(flight_info)

        logger.info(f"Found {len(flight_results)} flights")

        return FlightSearchResponse(
            success=True, flights=flight_results, total_results=len(flight_results)
        )

    except Exception as e:
        logger.error(f"Flight search error: {str(e)}")
        return FlightSearchResponse(
            success=False, flights=[], total_results=0, error=str(e)
        )


@app.get("/")
async def root():
    return {"message": "Flight Search API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
