from django.urls import path
from . import views

urlpatterns = [
    path('', views.accueil, name='accueil'),
    path('upload/', views.upload, name='upload'),
    path('labeler/', views.labeler, name='labeler'),
    path('labeler/<int:id>/', views.labelerId, name='labelerId'),
]
