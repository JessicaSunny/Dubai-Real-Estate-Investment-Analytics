# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 19:56:49 2026

@author: JR
"""

from pydantic import BaseModel

class PropertyInput(BaseModel):
    community: str
    property_type: str
    property_category: str
    view: str
    condition: str
    furnishing: str
    is_freehold: bool
    lat: float
    lon: float
    bedrooms: int
    area_sqft: float
    floor: float
    total_floors: float
    year_built: int
    parking_spaces: int
    chiller_included: bool
    metro_distance_min: int
    to_burj_khalifa_km: float
    mortgage_rate_at_listing: float