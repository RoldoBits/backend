//Connect to an api and get the data
function getData() {
    const response = await fetch('http://localhost:8000/')
    const data = await response.json()
    console.log(data)
}
