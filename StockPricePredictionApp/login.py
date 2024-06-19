from .models import StockUser
from django.shortcuts import render,redirect,HttpResponse
def user_login(request):
    database_password=""
    database_id=""
    database_email=""
    if request.method=='POST':
        email=request.POST['email']
        password=request.POST['password']
        print('email:',email)
        print('password:',password)
        user_data=userdata(email)
        for user_data in user_data:
            print(user_data)
            database_password=user_data.password
            database_id=user_data.id
            database_email=user_data.email
            #print('password: ',password)
            print('db_password: ',database_password)
        if password==database_password and email==database_email:
            #set_session_value=set_session(email)
            #print('session_value',set_session_value)
            set_session_id=request.session['id']=database_id
            set_session_email=request.session['email']=database_email
            print('set_session_email: ',set_session_email)
            print('set_session_id: ',set_session_id)
            #print('email: ',email)
            #return redirect('iris')
            #premsg='Hii Waiting...... now not complete code'
            #return render(request,'prediction.html')
            return redirect('home')
            #return render(request,'index.html',{'login_msg':'You are login',
            #                                        'session_msg':'Session is not set',
            #                                      'set_session_id':set_session_id,
            #                                     'set_session_email':set_session_email
                #                                   })
        
        else:
            msg="didn't match username and password"
            return render(request,'index.html',{'msg':msg})
            #return render(request,'index.html',{'msg':msg})
    else:
        msg='POST is mismatch'
        return render(request,'index.html',{'msg':msg})
        #return render(request,'Index.html')
#def set_session(self,email):
    #   request.session['email'] =email
    #  return request.session['email']
def userdata(email):
        #print('user_email: ',email)
        user_data=StockUser.objects.filter(email=email)
        return user_data