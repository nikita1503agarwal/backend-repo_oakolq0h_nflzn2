import os
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from bson import ObjectId
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

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


# Simple password hashing utilities (demo-grade)
_DEF_HASH_ITER = 100_000


def _hash_password(password: str, salt: str) -> str:
    data = (salt + password).encode("utf-8")
    # PBKDF2-HMAC with sha256 for better demo security without extra deps
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), _DEF_HASH_ITER).hex()


def _create_session(user_id: ObjectId) -> str:
    _ensure_db()
    token = secrets.token_urlsafe(32)
    session = {
        "user_id": user_id,
        "token": token,
        "created_at": datetime.now(timezone.utc),
        "expires_at": datetime.now(timezone.utc) + timedelta(days=7),
    }
    db["session"].insert_one(session)
    return token


def _get_user_from_token(auth_header: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Expect Authorization: Bearer <token>
    """
    if not auth_header:
        return None
    try:
        scheme, token = auth_header.split(" ", 1)
    except ValueError:
        return None
    if scheme.lower() != "bearer":
        return None
    _ensure_db()
    sess = db["session"].find_one({"token": token})
    if not sess:
        return None
    if sess.get("expires_at") and sess["expires_at"] < datetime.now(timezone.utc):
        return None
    user = db["user"].find_one({"_id": sess["user_id"]})
    return user


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


# ---------------------- AUTH ----------------------
class SignupPayload(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginPayload(BaseModel):
    email: EmailStr
    password: str


class AuthUser(BaseModel):
    id: str
    name: str
    email: EmailStr
    role: str
    is_active: bool = True


class AuthResponse(BaseModel):
    token: str
    user: AuthUser


@app.post("/api/auth/signup", response_model=AuthResponse)
def auth_signup(payload: SignupPayload):
    _ensure_db()
    existing = db["user"].find_one({"email": payload.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    salt = secrets.token_hex(16)
    pwd_hash = _hash_password(payload.password, salt)
    user_doc = {
        "name": payload.name,
        "email": payload.email,
        "role": "sales",
        "is_active": True,
        "password_salt": salt,
        "password_hash": pwd_hash,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    res = db["user"].insert_one(user_doc)
    token = _create_session(res.inserted_id)
    user_out = AuthUser(id=str(res.inserted_id), name=user_doc["name"], email=user_doc["email"], role=user_doc["role"], is_active=True)
    return {"token": token, "user": user_out}


@app.post("/api/auth/login", response_model=AuthResponse)
def auth_login(payload: LoginPayload):
    _ensure_db()
    user = db["user"].find_one({"email": payload.email})
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    salt = user.get("password_salt")
    if not salt:
        raise HTTPException(status_code=400, detail="Password not set for user")
    pwd_hash = _hash_password(payload.password, salt)
    if pwd_hash != user.get("password_hash"):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="User disabled")
    token = _create_session(user["_id"])
    user_out = AuthUser(id=str(user["_id"]), name=user["name"], email=user["email"], role=user.get("role", "sales"), is_active=user.get("is_active", True))
    return {"token": token, "user": user_out}


@app.get("/api/auth/me", response_model=AuthUser)
def auth_me(authorization: Optional[str] = Header(None)):
    _ensure_db()
    user = _get_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return AuthUser(id=str(user["_id"]), name=user["name"], email=user["email"], role=user.get("role", "sales"), is_active=user.get("is_active", True))


# -------------------- GENERIC LIST --------------------

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
            {"location": {"$regex": search, "$options": "i"}},
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


class CustomersBulkPayload(BaseModel):
    customers: List[Customer]


@app.post("/api/customers/bulk", status_code=201)
def create_customers_bulk(payload: CustomersBulkPayload):
    _ensure_db()
    # insert many with timestamps, similar to create_document
    docs = []
    from datetime import datetime, timezone
    for c in payload.customers:
        d = c.model_dump()
        d['created_at'] = datetime.now(timezone.utc)
        d['updated_at'] = datetime.now(timezone.utc)
        docs.append(d)
    result = db["customer"].insert_many(docs) if docs else None
    inserted = []
    if result and result.inserted_ids:
        for oid in result.inserted_ids:
            doc = db["customer"].find_one({"_id": oid})
            inserted.append(_to_dict(doc))
    return {"inserted_count": len(inserted), "items": inserted}


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
