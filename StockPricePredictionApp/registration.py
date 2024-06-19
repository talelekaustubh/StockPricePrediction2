from .models import StockUser
from django.db import IntegrityError
from django.shortcuts import render
from django.http import HttpResponse
from .varifyemail import mail_task

def user_register(request):
    if request.method=='POST':
        name=request.POST['name']
        email=request.POST['email']
        password=request.POST['password']
        confirm_password=request.POST['confirm_password']
        flag=0
        print(flag)
        print('email: ',email)
        varify_email_msg=mail_task(email)
        print('varify_email_msg:',varify_email_msg)
        if varify_email_msg:
            flag+=1
            print('inc_flag:',flag)
        else:
            flag=0
            print('flag_else:',flag)
        if password==confirm_password:
            try:
                ins=StockUser(name=name, email=email, password=password,flag=flag)
                if flag==1:
                    querys=ins.save()
                    querys
                    msg_sucess='you have sucessfully register'
                    return render(request,'index.html',{'msg_sucess':msg_sucess})
                    #return render(request,'index.html',{'msg_sucess':msg_sucess})
                    #return redirect('user_register')
                else:
                    msg_sucess='Please enter google verify email'
                    return render(request,'index.html',{'msg_sucess':msg_sucess})
            except IntegrityError as e:
                # Check if the exception is due to a UNIQUE constraint violation
                if 'UNIQUE constraint failed' in str(e):
                    email_unique="Email address already exists. Please choose a different email."
                    return render(request,'index.html',{'msg':email_unique})
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            pass_msg="didn't match password"
            return render(request,'index.html',{'msg':pass_msg})
            #return render(request,'home.html',{'pass_msg':"didn't match password"})        
    else:
        var_msg='failed!'
        return render(request,'index.html',{'msg':var_msg})
        #return render(request,'index.html',{'var_msg':'failed!'})
    
