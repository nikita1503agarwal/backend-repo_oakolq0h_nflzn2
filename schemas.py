"""
Database Schemas for CRM

Each Pydantic model represents a collection in MongoDB.
Collection name is the lowercase of the class name (e.g., Customer -> "customer").
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Literal
from datetime import datetime

# Core entities

class User(BaseModel):
    name: str = Field(..., description="Full name")
    email: EmailStr = Field(..., description="Email address")
    role: Literal["admin", "manager", "sales", "support", "marketing"] = "sales"
    is_active: bool = True

class Customer(BaseModel):
    name: str = Field(..., description="Customer full name or company contact")
    company: Optional[str] = Field(None, description="Company name")
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    stage: Literal["prospect", "active", "churn_risk", "inactive"] = "prospect"
    assigned_to: Optional[str] = Field(None, description="Owner user id or email")
    # Location fields
    location: Optional[str] = Field(None, description="City/State/Country or general location text")
    latitude: Optional[float] = Field(None, description="Latitude for mapping")
    longitude: Optional[float] = Field(None, description="Longitude for mapping")

class Lead(BaseModel):
    name: str
    email: Optional[EmailStr] = None
    source: Optional[str] = Field(None, description="Lead source e.g. Website, Referral")
    status: Literal["new", "contacted", "qualified", "won", "lost"] = "new"
    value: Optional[float] = Field(None, ge=0)
    owner: Optional[str] = None

class Activity(BaseModel):
    type: Literal["call", "email", "meeting", "note", "task"]
    subject: str
    notes: Optional[str] = None
    when: Optional[datetime] = None
    related_customer_id: Optional[str] = None
    related_lead_id: Optional[str] = None
    owner: Optional[str] = None

class Ticket(BaseModel):
    subject: str
    description: Optional[str] = None
    status: Literal["open", "in_progress", "resolved", "closed"] = "open"
    priority: Literal["low", "medium", "high", "urgent"] = "medium"
    customer_id: Optional[str] = None
    assignee: Optional[str] = None
