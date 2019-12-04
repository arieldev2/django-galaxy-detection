from django.urls import path, include
from .views import DetectionListView, DetectionDetailView, DetectionCreateView, DetectionUpdateView, DetectionDeleteView


urlpatterns = [
	path('', DetectionListView.as_view(), name="list"),
	path('create/', DetectionCreateView.as_view(), name="create"),
	path('detail/<int:pk>/', DetectionDetailView.as_view(), name="detail"),
	path('update/<int:pk>/', DetectionUpdateView.as_view(), name="update"),
	path('delete/<int:pk>/', DetectionDeleteView.as_view(), name="delete"),





    
]
