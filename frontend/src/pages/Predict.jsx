import React, { useState, useEffect } from 'react'

function Predict () {

    const [url, setUrl] = useState()
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)

    async function handleSubmit(e) {
        
        setLoading(true)
        e.preventDefault()

        try {
            const newsForm = await fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                headers: {
                    'content-type': 'application/JSON'
                },
                body: JSON.stringify({article_url: url})
            })

            if (newsForm.ok) {
                const data = await newsForm.json()
                setResult(data)
            } else {
                setResult(null)
                const errorData = await newsForm.json()
                setError(errorData.detail)
            }
        } catch {
            setError('500 Internal Server Error')
        } finally {
            setLoading(false)
        }
    }

    return (

        <>
        
            {result ? (
                <>
                    <h1>{result.title}</h1>
                    <h3>{result.prediction}</h3>
                    {result.prediction === "Real news" && (
                        <a href={url} target='_blank' rel='noopener noreferrer'>Access this article</a>
                    )}
                    <button onClick={() => setResult(null)}>Check another article</button>
                </>
            ) : 
                <form onSubmit={handleSubmit}>
                    <input value={url} onChange={(e) => setUrl(e.target.value)}></input>
                    <button type='submit'>Submit</button>
                </form>
            }
        
        </>
    )
}

export default Predict;