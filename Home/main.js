
// JavaScript to hide the preloader after 5 seconds
    window.addEventListener('load', function() {
        setTimeout(function() {
            document.getElementById('preloader').style.display = 'none';
            document.getElementById('main-content').style.display = 'block';
        }, 2000); // 5000 milliseconds = 5 seconds
    });


const navMenu = document.getElementById('nav-menu'),
    navToggle = document.getElementById('nav-toggle'),
    navClose = document.getElementById('nav-close')


if(navToggle){
    navToggle.addEventListener('click', () =>{
        navMenu.classList.add('show-menu')
    })
}


if(navClose){
    navClose.addEventListener('click', () =>{
        navMenu.classList.remove('show-menu')
    })
}

const navLink = document.querySelectorAll('.nav__link')

const linkAction = () =>{
    const navMenu = document.getElementById('nav-menu')

    navMenu.classList.remove('show-menu')
}
navLink.forEach(n => n.addEventListener('click', linkAction))


const scrollHeader = () =>{
    const header = document.getElementById('header')
    this.scrollY >= 50 ? header.classList.add('header-bg')
                        :header.classList.remove('header-bg')

}

window.addEventListener('scroll', scrollHeader)


const sr = ScrollReveal({
    origin: 'button',
    distance: '60px',
    duration: 2500
})

sr.reveal(`.home__images`, { distance: '120px', delay: 400 })
sr.reveal(`.home__title`, { delay: 1000 })
sr.reveal(`.home__description`, { delay: 1100 })
sr.reveal(`.home__button`, { delay: 1200 })
sr.reveal(`.home__footer`, { delay: 1300 })
