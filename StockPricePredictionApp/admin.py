from django.contrib import admin
from .models import StockUser
# Register your models here.
class StockAdmin(admin.ModelAdmin):
    list_display=('id','name','email','password','flag')
admin.site.register(StockUser,StockAdmin)
    