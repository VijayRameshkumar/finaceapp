from pydantic import BaseModel, conint

class VesselBase(BaseModel):
    cost_center: str
    imo_no: str
    vessel_code: str
    vessel_name: str
    vessel_type: str
    vessel_subtype: str
    build_year: conint(ge=1800, le=2500)

class VesselFilterParams(BaseModel):
    vessel_type: str
    vessel_subtype: list
    vessel_age_start: int
    vessel_age_end: int
    vessel_cat: list
    vessel_subcat: list
    selected_vessels_dropdown: list