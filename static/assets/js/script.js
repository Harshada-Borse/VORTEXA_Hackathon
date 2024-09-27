/*const container = document.querySelector('.container');
const LoginLink=document.querySelector('.SignInLink');
const RegisterLink=document.querySelector('.SignUpLink');
RegisterLink.addEventListener('click',()=>{
    container.classList.add('active;')
}) */

const container = document.querySelector('.container');
const signupLink = document.querySelector('.SignupLink');
const signinLink = document.querySelector('.SignInLink');

signupLink.addEventListener('click', () => {
    container.classList.add('active');
});

signinLink.addEventListener('click', () => {
    container.classList.remove('active');
});  

document.querySelector('.form-box.Register form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form from actually submitting
    
    // Hide the registration form
    document.querySelector('.container').style.display = 'none';
    
    // Show the thank you message
    document.getElementById('thankYouMessage').style.display = 'block';

});



