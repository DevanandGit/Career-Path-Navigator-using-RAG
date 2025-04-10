from django.urls import path
from . import views

urlpatterns = [
    path('whatsapp-webhook/', views.whatsapp_webhook, name='whatsapp-webhook'),
    path('send-message/', views.send_message, name='send-message'),  # Link to a view instead of the function
]
