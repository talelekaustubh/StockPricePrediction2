# views.py

from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.http import HttpResponse

def verify_email(request, token):
    try:
        # Find the user with the given token
        user = User.objects.get(profile__verification_token=token, profile__is_email_verified=False)
        
        # Mark the user's email as verified
        user.profile.is_email_verified = True
        user.profile.save()

        # You may want to log the user in automatically after email verification
        # Example: login(request, user)

        return HttpResponse("Email verified successfully!")
    except User.DoesNotExist:
        return HttpResponse("Invalid or expired token.")
