"""
Маршруты для аутентификации и авторизации пользователей.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User as UserModel
from app.schemas import UserCreate, User, Token, TokenData, ChangePasswordRequest

# Настройка логирования
logger = logging.getLogger(__name__)

# Конфигурация безопасности
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Инициализация хеширования паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Создание роутера
router = APIRouter(tags=["auth"])

# Функции безопасности
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db: Session, username: str):
    return db.query(UserModel).filter(UserModel.username == username).first()

def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_admin(current_user: UserModel = Depends(get_current_user)):
    """
    Проверяет, является ли текущий пользователь администратором.
    Если нет - возвращает 403 Forbidden.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав для доступа. Требуется роль администратора."
        )
    return current_user

# Маршруты

@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Маршрут для получения JWT токена.
    Принимает форму с именем пользователя и паролем.
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверное имя пользователя или пароль",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/register", response_model=User)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """
    Маршрут для регистрации нового пользователя.
    Принимает логин и пароль.
    """
    db_user = get_user(db, username=user.username)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пользователь уже существует"
        )
    hashed_password = get_password_hash(user.password)
    db_user = UserModel(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.get("/me", response_model=User)
async def read_users_me(current_user: UserModel = Depends(get_current_user)):
    """
    Возвращает информацию о текущем авторизованном пользователе.
    """
    return current_user

@router.post("/users/{username}/role", response_model=User)
async def update_user_role(
    username: str, 
    role: str,
    current_admin: UserModel = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Маршрут для обновления роли пользователя.
    Доступен только администратору.
    
    Args:
        username: имя пользователя, которому меняется роль
        role: новая роль (user/admin)
    """
    if role not in ["user", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Недопустимая роль. Допустимые значения: user, admin"
        )
    
    user = get_user(db, username=username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Пользователь {username} не найден"
        )
    
    user.role = role
    db.commit()
    db.refresh(user)
    return user

@router.post("/change-password")
async def change_password(
    password_data: ChangePasswordRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Маршрут для изменения пароля текущего пользователя.
    
    Args:
        current_password: текущий пароль пользователя
        new_password: новый пароль пользователя
    """
    # Проверяем текущий пароль
    if not verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Неверный текущий пароль"
        )
    
    # Хешируем и сохраняем новый пароль
    current_user.hashed_password = get_password_hash(password_data.new_password)
    db.commit()
    
    return {"status": "success", "message": "Пароль успешно изменен"}

@router.post("/users/{username}/reset-password")
async def reset_user_password(
    username: str,
    new_password: str,
    current_admin: UserModel = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Маршрут для сброса пароля пользователя администратором.
    Доступен только администратору.
    
    Args:
        username: имя пользователя, которому меняется пароль
        new_password: новый пароль пользователя
    """
    user = get_user(db, username=username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Пользователь {username} не найден"
        )
    
    # Хешируем и сохраняем новый пароль
    user.hashed_password = get_password_hash(new_password)
    db.commit()
    
    return {"status": "success", "message": f"Пароль пользователя {username} успешно изменен"}

@router.get("/users", response_model=list[User])
async def get_all_users(
    current_admin: UserModel = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Маршрут для получения списка всех пользователей.
    Доступен только администратору.
    """
    users = db.query(UserModel).all()
    return users

@router.delete("/users/{username}", response_model=dict)
async def delete_user(
    username: str,
    current_admin: UserModel = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Маршрут для удаления пользователя.
    Доступен только администратору.
    Администратор не может удалить себя.
    
    Args:
        username: имя пользователя, которого нужно удалить
    """
    # Проверяем, что администратор не пытается удалить себя
    if username == current_admin.username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Невозможно удалить свою учетную запись"
        )
    
    # Находим пользователя
    user = get_user(db, username=username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Пользователь {username} не найден"
        )
    
    # Удаляем пользователя
    db.delete(user)
    db.commit()
    
    return {"status": "success", "message": f"Пользователь {username} успешно удален"} 