from django.urls import path
from . import views

urlpatterns=[
    path('',views.index,name='index'),
    path('user_login/',views.user_login,name='user_login'),
    path('user_register/',views.user_register,name='user_register'),
    path('home/',views.home,name='home'),
    path('compare/',views.compare,name='compare'),
    path('download/<id>',views.download,name='download'),
    path('predict/',views.predict,name='predict'),
    path('all_stocks/',views.all_stocks,name='all_stocks'),
    path('details/<id>',views.details,name='details'),
    path('companycodesearch',views.companycodesearch,name='companycodesearch'),
    
]