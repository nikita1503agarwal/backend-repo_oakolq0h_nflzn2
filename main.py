import os
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bson import ObjectId

from database import db, create_document, get_documents
from schemas import Customer, Lead, Activity, Ticket, User

app = FastAPI(title="CRM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Utilities
class ObjectIdStr(BaseModel):
    id: str


def _ensure_db():
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")


def _to_dict(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not doc:
        return doc
    d = {**doc}
    _id = d.pop("_id", None)
    if isinstance(_id, ObjectId):
        d["id"] = str(_id)
    return d


@app.get("/")
def read_root():
    return {"message": "CRM Backend running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = getattr(db, "name", None) or "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    return response


# Generic list endpoint helper

def list_collection(
    collection: str,
    search: Optional[str] = None,
    limit: int = 50,
    skip: int = 0,
    extra_filter: Optional[Dict[str, Any]] = None,
):
    _ensure_db()
    filt: Dict[str, Any] = extra_filter.copy() if extra_filter else {}
    if search:
        # simple regex search across common fields
        filt["$or"] = [
            {"name": {"$regex": search, "$options": "i"}},
            {"email": {"$regex": search, "$options": "i"}},
            {"subject": {"$regex": search, "$options": "i"}},
            {"company": {"$regex": search, "$options": "i"}},
        ]
    cursor = db[collection].find(filt).skip(skip).limit(limit).sort("updated_at", -1)
    return [_to_dict(doc) for doc in cursor]


# Customers
@app.get("/api/customers")
def get_customers(
    search: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
):
    return list_collection("customer", search=search, limit=limit, skip=skip)


@app.post("/api/customers", status_code=201)
def create_customer(payload: Customer):
    _ensure_db()
    new_id = create_document("customer", payload)
    doc = db["customer"].find_one({"_id": ObjectId(new_id)})
    return _to_dict(doc)


# Leads
@app.get("/api/leads")
def get_leads(
    search: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
    status: Optional[str] = Query(None),
):
    extra = {"status": status} if status else None
    return list_collection("lead", search=search, limit=limit, skip=skip, extra_filter=extra)


@app.post("/api/leads", status_code=201)
def create_lead(payload: Lead):
    _ensure_db()
    new_id = create_document("lead", payload)
    doc = db["lead"].find_one({"_id": ObjectId(new_id)})
    return _to_dict(doc)


# Activities
@app.get("/api/activities")
def get_activities(
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
    owner: Optional[str] = Query(None),
):
    extra = {"owner": owner} if owner else None
    return list_collection("activity", limit=limit, skip=skip, extra_filter=extra)


@app.post("/api/activities", status_code=201)
def create_activity(payload: Activity):
    _ensure_db()
    new_id = create_document("activity", payload)
    doc = db["activity"].find_one({"_id": ObjectId(new_id)})
    return _to_dict(doc)


# Tickets
@app.get("/api/tickets")
def get_tickets(
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
):
    extra = {"status": status} if status else None
    return list_collection("ticket", limit=limit, skip=skip, extra_filter=extra)


@app.post("/api/tickets", status_code=201)
def create_ticket(payload: Ticket):
    _ensure_db()
    new_id = create_document("ticket", payload)
    doc = db["ticket"].find_one({"_id": ObjectId(new_id)})
    return _to_dict(doc)


# Simple analytics
@app.get("/api/analytics/overview")
def analytics_overview():
    _ensure_db()
    totals = {
        "customers": db["customer"].count_documents({}),
        "leads": db["lead"].count_documents({}),
        "open_tickets": db["ticket"].count_documents({"status": {"$in": ["open", "in_progress"]}}),
        "activities": db["activity"].count_documents({}),
    }

    # pipeline by status for leads
    pipeline = [
        {"$group": {"_id": "$status", "count": {"$sum": 1}, "value": {"$sum": {"$ifNull": ["$value", 0]}}}},
        {"$project": {"status": "$_id", "count": 1, "value": 1, "_id": 0}},
    ]
    by_status = list(db["lead"].aggregate(pipeline))

    return {"totals": totals, "pipeline": by_status}


# Webhook placeholder (integration point)
class WebhookPayload(BaseModel):
    provider: str
    event: str
    data: dict


@app.post("/api/integrations/webhook")
def receive_webhook(payload: WebhookPayload):
    _ensure_db()
    # store webhook events for auditing
    _id = create_document("webhook", payload.model_dump())
    return {"received": True, "id": _id}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
